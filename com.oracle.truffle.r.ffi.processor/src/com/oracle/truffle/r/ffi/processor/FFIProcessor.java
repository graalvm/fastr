/*
 * Copyright (c) 2017, 2020, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 3 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 3 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 3 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package com.oracle.truffle.r.ffi.processor;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.io.Writer;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.RoundEnvironment;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.MirroredTypeException;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.Types;
import javax.tools.Diagnostic;
import javax.tools.Diagnostic.Kind;

import javax.tools.JavaFileObject;

public final class FFIProcessor extends AbstractProcessor {

    @Override
    public Set<String> getSupportedAnnotationTypes() {
        Set<String> annotations = new HashSet<>();
        annotations.add("com.oracle.truffle.r.ffi.processor.RFFIUpCallRoot");
        return annotations;
    }

    @Override
    public SourceVersion getSupportedSourceVersion() {
        return SourceVersion.latest();
    }

    @Override
    public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
        if (roundEnv.processingOver()) {
            return false;
        }
        process0(roundEnv);
        return true;
    }

    private void process0(RoundEnvironment roundEnv) {
        for (Element e : roundEnv.getElementsAnnotatedWith(RFFIUpCallRoot.class)) {
            try {
                processElement(e);
            } catch (Throwable ex) {
                ex.printStackTrace();
                String message = "Uncaught error in " + this.getClass();
                processingEnv.getMessager().printMessage(Kind.ERROR, message + ": " + printException(ex), e);
            }
        }
    }

    private void processElement(Element e) throws IOException {
        if (e.getKind() != ElementKind.INTERFACE) {
            processingEnv.getMessager().printMessage(Kind.ERROR, "RFFIUpCallRoot must annotate an interface");
        }
        Types types = processingEnv.getTypeUtils();
        TypeElement typeElement = (TypeElement) e;
        List<? extends TypeMirror> extended = typeElement.getInterfaces();
        int count = 0;
        for (TypeMirror tm : extended) {
            TypeElement x = (TypeElement) types.asElement(tm);
            List<? extends Element> methods = x.getEnclosedElements();
            count += methods.size();
        }
        ExecutableElement[] methods = new ExecutableElement[count];
        count = 0;
        for (TypeMirror tm : extended) {
            TypeElement x = (TypeElement) types.asElement(tm);
            List<? extends Element> encMethods = x.getEnclosedElements();
            for (Element encMethod : encMethods) {
                methods[count++] = (ExecutableElement) encMethod;
            }
        }
        Arrays.sort(methods, new Comparator<ExecutableElement>() {
            @Override
            public int compare(ExecutableElement e1, ExecutableElement e2) {
                return e1.getSimpleName().toString().compareTo(e2.getSimpleName().toString());
            }
        });
        generateTable(methods);
        generateMessageClasses(methods);
        generateCallbacks(methods);
        note("If you edited any UpCallsRFFI interfaces do: 'mx rfficodegen'.\n");
    }

    private void generateTable(ExecutableElement[] methods) throws IOException {
        JavaFileObject fileObj = processingEnv.getFiler().createSourceFile("com.oracle.truffle.r.ffi.impl.upcalls.RFFIUpCallTable");
        Writer w = fileObj.openWriter();
        w.append("// GENERATED by com.oracle.truffle.r.ffi.processor.FFIProcessor class; DO NOT EDIT\n");
        w.append("package com.oracle.truffle.r.ffi.impl.upcalls;\n");
        w.append("/**\n" +
                        " * The following enum contains an entry for each method in the interface annoated\n" +
                        " * with @RFFIUpCallRoot including inherited methods.\n" +
                        " */");
        w.append("public enum RFFIUpCallTable {\n");
        for (int i = 0; i < methods.length; i++) {
            ExecutableElement method = methods[i];
            w.append("    ").append(method.getSimpleName().toString()).append(i == methods.length - 1 ? ";" : ",").append('\n');
        }

        w.append("}\n");
        w.close();
    }

    private void generateMessageClasses(ExecutableElement[] methods) throws IOException {
        for (int i = 0; i < methods.length; i++) {
            ExecutableElement m = methods[i];
            generateCallClass(m);
        }
    }

    private void generateCallClass(ExecutableElement m) throws IOException {
        RFFIUpCallNode nodeAnnotation = m.getAnnotation(RFFIUpCallNode.class);
        String canRunGc = m.getAnnotation(RFFIRunGC.class) == null ? "false" : "true";
        boolean needsNode = false;
        boolean nodeHasCreate = false;
        TypeElement nodeClass = null;
        String nodeClassName = null;
        String nodeQualifiedClassName = null;
        boolean needsFrame = false;
        boolean needsCallTarget = false;
        boolean generateCreateNodeMethod = false;
        boolean needsFunction = false;
        boolean functionHasCreate = false;
        TypeElement functionClass = null;
        String functionClassName = null;
        String functionQualifiedClassName = null;

        if (nodeAnnotation != null) {
            try {
                nodeAnnotation.value();
            } catch (MirroredTypeException e) {
                nodeClass = (TypeElement) processingEnv.getTypeUtils().asElement(e.getTypeMirror());
                if (nodeClass != null) {
                    nodeQualifiedClassName = nodeClass.getQualifiedName().toString();
                    nodeClassName = nodeQualifiedClassName.substring(nodeQualifiedClassName.lastIndexOf(".") + 1);
                    needsNode = true;
                }
            }
            try {
                nodeAnnotation.functionClass();
            } catch (MirroredTypeException e) {
                functionClass = (TypeElement) processingEnv.getTypeUtils().asElement(e.getTypeMirror());
                functionQualifiedClassName = functionClass.getQualifiedName().toString();
                if (!Void.class.getName().equals(functionQualifiedClassName)) {
                    needsFunction = true;
                    functionClassName = functionQualifiedClassName.substring(functionQualifiedClassName.lastIndexOf(".") + 1);
                } else {
                    functionClass = null;
                    functionQualifiedClassName = null;
                }
            }
            if (needsNode) {
                needsCallTarget = nodeAnnotation.needsCallTarget();
            }
        }
        if (needsNode) {
            nodeHasCreate = hasCreate(nodeClass);
            if (needsFunction) {
                functionHasCreate = hasCreate(functionClass);
            }
            needsFrame = needsFrame(nodeClass);
        }

        // process arguments first to see if unwrap is necessary
        List<? extends VariableElement> params = m.getParameters();
        StringBuilder arguments = new StringBuilder();
        StringBuilder unwrapNodes = new StringBuilder();
        StringBuilder unwrappedArgs = new StringBuilder();
        boolean needsUnwrapImport = false;
        boolean needsStringUnwrapImport = false;
        Set<String> unwrapArrayImports = new HashSet<>();
        int injectedArgsCount = 0;

        if (!needsCallTarget && needsFunction) {
            arguments.append("function, ");
        }
        for (int i = 0; i < params.size(); i++) {
            if (i != 0) {
                arguments.append(", ");
            }
            TypeMirror paramType = params.get(i).asType();

            RFFICpointer[] pointerAnnotations = params.get(i).getAnnotationsByType(RFFICpointer.class);
            RFFICarray[] arrayAnnotations = params.get(i).getAnnotationsByType(RFFICarray.class);
            RFFIInject[] injectAnnotations = params.get(i).getAnnotationsByType(RFFIInject.class);
            boolean needsArrayUnwrap = arrayAnnotations.length > 0;

            String paramName = params.get(i).getSimpleName().toString();
            String paramTypeName = getTypeName(paramType);
            boolean needsInject = injectAnnotations.length > 0;
            boolean needsUnwrap = !needsInject &&
                            !paramType.getKind().isPrimitive() &&
                            !paramTypeName.equals("java.lang.String") &&
                            pointerAnnotations.length == 0 &&
                            !needsArrayUnwrap;
            boolean needsStringUnwrap = paramTypeName.equals("java.lang.String");
            boolean needCast = !paramTypeName.equals("java.lang.Object");
            if (needCast) {
                arguments.append('(').append(paramTypeName).append(") ");
            }
            needsUnwrapImport |= needsUnwrap;
            needsStringUnwrapImport |= needsStringUnwrap;
            unwrappedArgs.append("        final Object ").append(paramName).append("Unwrapped = ");
            if (needsInject) {
                if (nodeAnnotation != null) {
                    // TODO: report an error, injections are for non-node invocations only
                }
                if (paramTypeName.equals("com.oracle.truffle.r.runtime.context.RContext")) {
                    unwrappedArgs.append("ctx");
                } else {
                    // TODO: report an error
                }
                injectedArgsCount++;
            } else if (needsUnwrap) {
                unwrapNodes.append("                @Cached() FFIUnwrapNode ").append(paramName).append("Unwrap,\n");
                unwrappedArgs.append(paramName).append("Unwrap").append(".execute(");
            } else if (needsArrayUnwrap) {
                RFFICarray arrAnnot = arrayAnnotations[0];
                processingEnv.getMessager().printMessage(Kind.NOTE, "Element type: ");
                processingEnv.getMessager().printMessage(Kind.NOTE, "" + arrAnnot.element());
                String arrayUnwrapSimpleName = arrAnnot.element().wrapperSimpleClassName;
                unwrapArrayImports.add(arrAnnot.element().wrapperClassPackage + "." + arrayUnwrapSimpleName);
                unwrapNodes.append("                @Cached ").append(arrayUnwrapSimpleName).append(' ').append(paramName).append("Unwrap,\n");
                unwrappedArgs.append(paramName).append("Unwrap").append(".execute(");
                if (!"".equals(arrAnnot.length())) {
                    String lengthExpr = arrAnnot.length();
                    lengthExpr = lengthExpr.replaceAll("\\{(.*?)\\}", "(int) arg$1");
                    unwrappedArgs.append(lengthExpr);
                } else {
                    // TODO: report an error
                }
                unwrappedArgs.append(", ");
            } else if (needsStringUnwrap) {
                unwrapNodes.append("                @Cached FFIUnwrapString ").append(paramName).append("Unwrap,\n");
                unwrappedArgs.append(paramName).append("Unwrap").append(".execute(");
            }
            if (!needsInject) {
                unwrappedArgs.append("arguments[").append(i).append("]");
                arguments.append(paramName).append("Unwrapped");
                if (needsUnwrap || needsArrayUnwrap || needsStringUnwrap) {
                    unwrappedArgs.append(')');
                }
            } else {
                arguments.append(paramName).append("Unwrapped");
            }
            unwrappedArgs.append(";\n");
            // add an indexed copy of the unwrapped parameter for a potential referencing from
            // an array length expression in RFFICarray
            unwrappedArgs.append("        final Object arg").append(i).append(" = ").append(paramName).append("Unwrapped;\n");

        }

        TypeKind returnKind = m.getReturnType().getKind();
        boolean needsReturnArrayWrap = m.getAnnotationsByType(RFFICarray.class).length > 0;
        boolean needsReturnWrap = (returnKind != TypeKind.VOID && !returnKind.isPrimitive() &&
                        !"java.lang.String".equals(getTypeName(m.getReturnType())) &&
                        m.getAnnotationsByType(RFFICpointer.class).length == 0) ||
                        needsReturnArrayWrap;

        if (needsReturnWrap) {
            unwrapNodes.append("                @Cached() FFIMaterializeNode materializeNode,\n");
            unwrapNodes.append("                @Cached() FFIToNativeMirrorNode toNativeWrapperNode,\n");
        }

        String name = m.getSimpleName().toString();
        String callName = name + "Call";
        JavaFileObject fileObj = processingEnv.getFiler().createSourceFile("com.oracle.truffle.r.ffi.impl.upcalls." + callName);
        Writer w = fileObj.openWriter();
        w.append("// GENERATED by com.oracle.truffle.r.ffi.processor.FFIProcessor class; DO NOT EDIT\n");
        w.append("\n");
        w.append("package com.oracle.truffle.r.ffi.impl.upcalls;\n");
        w.append("\n");
        w.append("\n");
        w.append("import com.oracle.truffle.api.CompilerDirectives;\n");
        if (!returnKind.isPrimitive() && returnKind != TypeKind.VOID) {
            w.append("import com.oracle.truffle.r.runtime.data.RDataFactory;\n");
        }
        w.append("import com.oracle.truffle.r.ffi.impl.upcalls.UpCallBase;");
        w.append("import com.oracle.truffle.r.runtime.ffi.RFFIContext;\n");
        w.append("import com.oracle.truffle.r.runtime.ffi.RFFILog;\n");
        w.append("import com.oracle.truffle.api.interop.InteropLibrary;\n");
        w.append("import com.oracle.truffle.api.library.ExportLibrary;\n");
        w.append("import com.oracle.truffle.api.library.ExportMessage;\n");
        w.append("import com.oracle.truffle.api.nodes.ControlFlowException;\n");
        w.append("import com.oracle.truffle.api.dsl.Cached;\n");
        w.append("import com.oracle.truffle.api.profiles.ValueProfile;\n");
        w.append("import com.oracle.truffle.r.ffi.impl.upcalls.UpCallsRFFI.HandleUpCallExceptionNode;\n");
        w.append("import com.oracle.truffle.r.runtime.RError;\n");
        w.append("import com.oracle.truffle.r.runtime.context.RContext;\n");
        if (needsNode) {
            w.append("import ").append(nodeQualifiedClassName).append(";\n");
        }
        if (needsFunction) {
            w.append("import ").append(functionQualifiedClassName).append(";\n");
        }
        w.append("import com.oracle.truffle.api.profiles.BranchProfile;\n");
        if (needsCallTarget) {
            w.append("import com.oracle.truffle.api.nodes.RootNode;\n");
            w.append("import com.oracle.truffle.api.frame.VirtualFrame;\n");
            w.append("import com.oracle.truffle.api.Truffle;\n");
            w.append("import com.oracle.truffle.api.CallTarget;\n");
            w.append("import com.oracle.truffle.r.runtime.context.RFFIUpCallTargets;\n");
        }
        if (needsUnwrapImport) {
            w.append("import com.oracle.truffle.r.runtime.ffi.FFIUnwrapNode;\n");
        }
        if (needsStringUnwrapImport) {
            w.append("import com.oracle.truffle.r.runtime.ffi.FFIUnwrapString;\n");
        }
        if (!unwrapArrayImports.isEmpty()) {
            for (String unwrapArrayImport : unwrapArrayImports) {
                w.append("import ").append(unwrapArrayImport).append(";\n");
            }
        }
        if (needsReturnWrap) {
            w.append("import com.oracle.truffle.r.runtime.ffi.FFIWrap.FFIUpCallWrap;\n");
            w.append("import com.oracle.truffle.r.runtime.ffi.FFIWrap.FFIUpCallWrap.FFIWrapResult;\n");
            w.append("import com.oracle.truffle.r.runtime.ffi.FFIMaterializeNode;\n");
            w.append("import com.oracle.truffle.r.runtime.ffi.FFIToNativeMirrorNode;\n");
        }
        w.append("import com.oracle.truffle.api.TruffleLanguage.ContextReference;\n");
        w.append("import com.oracle.truffle.api.dsl.CachedContext;\n");
        w.append("import com.oracle.truffle.r.runtime.context.TruffleRLanguage;\n");

        w.append("\n");

        w.append("// Checkstyle: stop method name check\n");
        w.append("@ExportLibrary(InteropLibrary.class)\n");
        w.append("final class ").append(callName).append(" extends UpCallBase {\n");
        w.append('\n');
        w.append("    protected final UpCallsRFFI upCallsImpl;\n");
        w.append("    ").append(callName).append("(UpCallsRFFI upCallsImpl) {\n");
        w.append("        assert upCallsImpl != null;\n");
        w.append("        this.upCallsImpl = upCallsImpl;\n");
        w.append("    }\n");
        w.append("\n");
        w.append("    @ExportMessage\n");
        w.append("    @SuppressWarnings(\"static-method\")\n");
        w.append("    boolean isExecutable(){\n");
        w.append("        return true;\n");
        w.append("    }\n");
        w.append("\n");
        w.append("    @ExportMessage\n");
        w.append("    @SuppressWarnings({\"unused\", \"cast\"})\n");
        w.append("    Object execute(Object[] arguments,\n");
        if (unwrapNodes.length() > 0) {
            w.append(unwrapNodes);
        }

        w.append("                @CachedContext(TruffleRLanguage.class) ContextReference<RContext> ctxRef,\n");
        if (needsCallTarget) {
            w.append("                @Cached() com.oracle.truffle.r.ffi.impl.upcalls.UpCallBase.CallNode callNode,\n");
            w.append("                @Cached(value = \"createCallTarget(ctxRef)\", allowUncached = true) CallTarget callTarget,\n");
        } else if (needsNode) {
            if (nodeClass.getModifiers().contains(Modifier.ABSTRACT)) {
                w.append("                @Cached() " + nodeClassName + " node,\n");
                if (!nodeHasCreate) {
                    processingEnv.getMessager().printMessage(Kind.ERROR, "Need static create for abstract classes", m);
                }
            } else {
                w.append("                @Cached(\"createNode()\") " + nodeClassName + " node,\n");
                generateCreateNodeMethod = true;
            }
            if (needsFunction) {
                w.append("                @Cached() " + functionClassName + " function" + ",\n");
            }
        }

        w.append("                @Cached(\"createClassProfile()\") ValueProfile upCallProfile,\n");
        w.append("                @Cached(\"createClassProfile()\") ValueProfile ctxProfile,\n");
        w.append("                @Cached BranchProfile controlFlowExBranch,\n");
        w.append("                @Cached BranchProfile errorExBranch,\n");
        w.append("                @Cached(value = \"this.upCallsImpl.createHandleUpCallExceptionNode()\", uncached = \"this.upCallsImpl.getUncachedHandleUpCallExceptionNode()\") HandleUpCallExceptionNode handleExceptionNode");
        w.append(") {\n");
        w.append("\n");
        w.append("        assert arguments.length == " + (params.size() - injectedArgsCount) + " : \"wrong number of arguments passed to " + name + "\";\n");
        w.append("        if (RFFILog.logEnabled()) {\n");
        w.append("            RFFILog.logUpCall(\"" + name + "\", arguments);\n");
        w.append("        }\n");
        w.append("        RContext ctx = ctxRef.get();\n");
        w.append("        RFFIContext rffiCtx = ctxProfile.profile(ctx.getStateRFFI());\n");

        if (returnKind != TypeKind.VOID) {
            w.append("        Object resultRObj0;\n");
            w.append("        Object resultRObj;\n");
        }
        if (needsReturnWrap) {
            w.append("        Object registerRObj;\n");
        }
        w.append("        UpCallsRFFI impl = upCallProfile.profile(upCallsImpl);\n");
        w.append("        rffiCtx.beforeUpcall(ctx, " + canRunGc + ", impl.getRFFIType());\n");
        w.append(unwrappedArgs);
        w.append("        try {\n");

        w.append("            ");
        if (returnKind != TypeKind.VOID) {
            w.append("resultRObj0 = ");
        }
        if (needsNode) {
            if (needsCallTarget) {
                w.append("callNode.call");
            } else {
                w.append("node.executeObject");
            }
        } else {
            w.append("impl." + name);
        }
        w.append("(");
        if (needsCallTarget) {
            w.append("callTarget, ");
        }
        if (needsFrame) {
            w.append("frame, ");
        }
        w.append(arguments).append(");\n");

        if (returnKind != TypeKind.VOID) {
            if (needsReturnWrap) {
                w.append("            FFIWrapResult result = FFIUpCallWrap.wrap(resultRObj0, materializeNode, toNativeWrapperNode);\n");
                w.append("            resultRObj = result.nativeMirror;\n");
                w.append("            registerRObj = result.materialized;\n");
            } else {
                w.append("            resultRObj = resultRObj0;\n");
            }
        }
        w.append("        } catch (ControlFlowException ex) {\n");
        w.append("            controlFlowExBranch.enter();\n");
        w.append("            handleExceptionNode.execute(ex);\n");
        appendCreateDummyResultObj(returnKind, needsReturnWrap, w);
        w.append("        } catch (RError ex) {\n");
        w.append("            errorExBranch.enter();\n");
        w.append("            handleExceptionNode.execute(ex);\n");
        appendCreateDummyResultObj(returnKind, needsReturnWrap, w);
        w.append("        } catch (Throwable ex) {\n");
        w.append("            CompilerDirectives.transferToInterpreter();\n");
        w.append("            assert reportException(ex);\n");
        w.append("            RFFILog.logException(ex);\n");
        w.append("            handleExceptionNode.execute(ex);\n");
        appendCreateDummyResultObj(returnKind, needsReturnWrap, w);
        w.append("        }\n");
        w.append("        rffiCtx.afterUpcall(" + canRunGc + ", impl.getRFFIType());\n");
        if (returnKind == TypeKind.VOID) {
            w.append("        if (RFFILog.logEnabled()) {\n");
            w.append("            RFFILog.logUpCallReturn(\"" + name + "\", null);\n");
            w.append("        }\n");
            w.append("        return 0; // void return type\n");
        } else {
            if (!returnKind.isPrimitive() && m.getAnnotationsByType(RFFICpointer.class).length == 0) {
                w.append("        if (impl.getRFFIType() == com.oracle.truffle.r.runtime.ffi.RFFIFactory.Type.NFI) {\n");
                if (needsReturnWrap) {
                    w.append("            rffiCtx.registerReferenceUsedInNative(registerRObj); \n");
                } else {
                    w.append("            rffiCtx.registerReferenceUsedInNative(resultRObj); \n");
                }
                w.append("        }\n");
            }
            w.append("        if (RFFILog.logEnabled()) {\n");
            w.append("            RFFILog.logUpCallReturn(\"" + name + "\", resultRObj);\n");
            w.append("        }\n");
            w.append("        return resultRObj;\n");
        }
        w.append("    }\n");
        w.append("\n");

        if (needsCallTarget) {
            w.append("    protected static CallTarget createCallTarget(ContextReference<RContext> ctxRef) {\n");
            w.append("        RFFIUpCallTargets targets = ctxRef.get().getRFFIUpCallTargets();\n");
            w.append("        if(targets.").append(nodeClassName).append(" == null) {\n");
            w.append("            targets.").append(nodeClassName).append(" =  Truffle.getRuntime().createCallTarget(new NodeRootNode());\n");
            w.append("        }\n");
            w.append("        return targets.").append(nodeClassName).append(";\n");
            w.append("    }\n");
            w.append("\n");

            w.append("    private final static class NodeRootNode extends RootNode {\n");
            if (nodeClass.getModifiers().contains(Modifier.ABSTRACT)) {
                if (nodeHasCreate) {
                    w.append("        @Child private " + nodeClassName + " node = " + nodeClassName + ".create();\n");
                } else {
                    w.append("        @Child private " + nodeClassName + " node;\n");
                    processingEnv.getMessager().printMessage(Kind.ERROR, "Need static create for abstract classes", m);
                }
            } else {
                w.append("        @Child private " + nodeClassName + " node = new " + nodeClassName + "();\n");
            }

            if (needsFunction) {
                if (functionHasCreate) {
                    w.append("        @Child private " + functionClassName + " function = " + functionClassName + ".create();\n");
                } else {
                    w.append("        @Child private " + functionClassName + " function = new " + functionClassName + "();\n");
                }
            }

            w.append("\n");
            w.append("        public NodeRootNode() {\n");
            w.append("            super(null);\n");
            w.append("        }\n");
            w.append("\n");
            w.append("        @Override\n");
            w.append("        public Object execute(VirtualFrame frame) {\n");
            w.append("            Object[] arguments = frame.getArguments();\n");

            w.append("            return node.executeObject(");
            if (needsFrame) {
                w.append("frame, ");
            }

            StringBuilder args = new StringBuilder();
            if (needsFunction) {
                args.append("function, ");
            }
            for (int i = 0; i < params.size(); i++) {
                if (i != 0) {
                    args.append(", ");
                }
                TypeMirror paramType = params.get(i).asType();
                String paramTypeName = getTypeName(paramType);
                boolean needCast = !paramTypeName.equals("java.lang.Object") && !paramTypeName.equals("java.lang.String");
                if (needCast) {
                    args.append('(').append(paramTypeName).append(") ");
                }
                args.append("arguments[").append(i).append("]");
            }
            w.append(args).append(");\n");

            w.append("        }\n");
            w.append("    }\n");
        }

        if (generateCreateNodeMethod) {
            w.append("    protected static " + nodeClassName + " createNode() {\n");
            w.append("        return new " + nodeClassName + "();\n");
            w.append("    }\n");
        }
        w.append("}\n");
        w.close();
    }

    private static void appendCreateDummyResultObj(TypeKind returnKind, boolean needsReturnWrap, Writer w) throws IOException {
        if (returnKind.isPrimitive()) {
            w.append("            resultRObj = -1;\n");
        } else if (returnKind != TypeKind.VOID) {
            w.append("            resultRObj = RDataFactory.createIntVectorFromScalar(-1);\n");
            if (needsReturnWrap) {
                w.append("            registerRObj = resultRObj;\n");
            }
        }
    }

    private void generateCallbacks(ExecutableElement[] methods) throws IOException {
        JavaFileObject fileObj = processingEnv.getFiler().createSourceFile("com.oracle.truffle.r.ffi.impl.upcalls.Callbacks");
        Writer w = fileObj.openWriter();
        w.append("// GENERATED; DO NOT EDIT\n\n");
        w.append("package com.oracle.truffle.r.ffi.impl.upcalls;\n\n");
        w.append("import com.oracle.truffle.api.interop.TruffleObject;\n");
        w.append("import com.oracle.truffle.api.CompilerDirectives.CompilationFinal;\n");
        w.append("import com.oracle.truffle.r.ffi.impl.upcalls.UpCallsRFFI;\n\n");
        w.append("public enum Callbacks {\n");
        for (int i = 0; i < methods.length; i++) {
            ExecutableElement m = methods[i];
            String sig = getNFISignature(m);
            w.append("    ").append(m.getSimpleName().toString()).append('(').append('"').append(sig).append('"').append(')');
            w.append(i == methods.length - 1 ? ';' : ',');
            w.append("\n");
        }
        w.append('\n');
        w.append("    public final String nfiSignature;\n");
        w.append("    @CompilationFinal public TruffleObject call;\n\n");
        w.append("    Callbacks(String signature) {\n");
        w.append("        this.nfiSignature = signature;\n");
        w.append("    }\n\n");

        w.append("    public static void createCalls(UpCallsRFFI upCallsRFFIImpl) {\n");
        for (int i = 0; i < methods.length; i++) {
            ExecutableElement m = methods[i];
            String callName = m.getSimpleName().toString();
            w.append("        ").append(callName).append(".call = new ").append(callName).append("Call(upCallsRFFIImpl);\n");
        }
        w.append("    }\n");
        w.append("}\n");
        w.close();
    }

    private String getNFISignature(ExecutableElement m) {
        List<? extends VariableElement> params = m.getParameters();
        int lparams = params.size();
        StringBuilder sb = new StringBuilder();
        sb.append('(');
        int realArgsCounter = 0;
        for (int i = 0; i < lparams; i++) {
            VariableElement param = params.get(i);
            RFFIInject[] injectAnnotations = params.get(i).getAnnotationsByType(RFFIInject.class);
            if (injectAnnotations.length > 0) {
                continue;
            }
            if (realArgsCounter > 0) {
                sb.append(", ");
            }
            realArgsCounter++;
            String nfiParam = nfiParamName(param.asType(), false, param);
            sb.append(nfiParam);
        }
        sb.append(')');
        sb.append(" : ");
        sb.append(nfiParamName(m.getReturnType(), true, m));
        return sb.toString();
    }

    private String nfiParamName(TypeMirror paramType, boolean isReturn, Element m) {
        String paramTypeName = getTypeName(paramType);
        switch (paramTypeName) {
            case "java.lang.Object":
                return "pointer";
            case "boolean":
                return "uint8";
            case "int":
                return "sint32";
            case "long":
                return "sint64";
            case "double":
                return "double";
            case "void":
                return "void";
            case "int[]":
                return "[sint32]";
            case "long[]":
                return "[sint64]";
            case "double[]":
                return "[double]";
            case "byte[]":
                return "[uint8]";
            default:
                if ("java.lang.String".equals(paramTypeName)) {
                    return "string";
                }
                if (isReturn) {
                    processingEnv.getMessager().printMessage(Kind.ERROR, "Invalid return type " + paramTypeName, m);
                } else {
                    processingEnv.getMessager().printMessage(Kind.ERROR, "Invalid parameter type " + paramTypeName, m);
                }
                return "pointer";
        }
    }

    private String getTypeName(TypeMirror type) {
        Types types = processingEnv.getTypeUtils();
        TypeKind kind = type.getKind();
        String returnType;
        if (kind.isPrimitive() || kind == TypeKind.VOID) {
            returnType = kind.name().toLowerCase();
        } else {
            Element rt = types.asElement(type);
            returnType = rt.toString();
        }
        return returnType;
    }

    private void note(String msg) {
        processingEnv.getMessager().printMessage(Diagnostic.Kind.NOTE, msg);
    }

    private static String printException(Throwable e) {
        StringWriter string = new StringWriter();
        PrintWriter writer = new PrintWriter(string);
        e.printStackTrace(writer);
        writer.flush();
        string.flush();
        return e.getMessage() + "\r\n" + string.toString();
    }

    private static boolean hasCreate(TypeElement clazz) {
        if (clazz == null) {
            return false;
        }
        for (Element element : clazz.getEnclosedElements()) {
            if (element.getKind() == ElementKind.METHOD && element.getModifiers().contains(Modifier.STATIC) && "create".equals(element.getSimpleName().toString())) {
                return true;
            }
        }
        return false;
    }

    private static boolean needsFrame(TypeElement nodeClass) {
        for (Element element : nodeClass.getEnclosedElements()) {
            if (element.getKind() == ElementKind.METHOD && element.getModifiers().contains(Modifier.ABSTRACT) && "executeObject".equals(element.getSimpleName().toString())) {
                final List<? extends VariableElement> parameters = ((ExecutableElement) element).getParameters();
                if (!parameters.isEmpty()) {
                    final VariableElement first = parameters.get(0);
                    final TypeMirror parType = first.asType();
                    if (parType.toString().equals("com.oracle.truffle.api.frame.VirtualFrame")) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
}
