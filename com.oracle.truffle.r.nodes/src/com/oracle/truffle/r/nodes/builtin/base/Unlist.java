/*
 * Copyright (c) 2013, 2014, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package com.oracle.truffle.r.nodes.builtin.base;

import static com.oracle.truffle.r.runtime.RBuiltinKind.*;

import com.oracle.truffle.api.*;
import com.oracle.truffle.api.CompilerDirectives.SlowPath;
import com.oracle.truffle.api.dsl.*;
import com.oracle.truffle.api.frame.*;
import com.oracle.truffle.r.nodes.*;
import com.oracle.truffle.r.nodes.access.*;
import com.oracle.truffle.r.nodes.builtin.*;
import com.oracle.truffle.r.nodes.unary.*;
import com.oracle.truffle.r.runtime.*;
import com.oracle.truffle.r.runtime.data.*;
import com.oracle.truffle.r.runtime.data.model.*;
import com.oracle.truffle.r.nodes.builtin.base.UnlistFactory.RecursiveLengthFactory;

@RBuiltin(name = "unlist", kind = SUBSTITUTE)
// TODO INTERNAL
public abstract class Unlist extends RBuiltinNode {

    private static final Object[] PARAMETER_NAMES = new Object[]{"x", "recursive", "use.names"};

    @Override
    public Object[] getParameterNames() {
        return PARAMETER_NAMES;
    }

    @Override
    public RNode[] getParameterValues() {
        return new RNode[]{ConstantNode.create(RMissing.instance), ConstantNode.create(RRuntime.LOGICAL_TRUE), ConstantNode.create(RRuntime.LOGICAL_TRUE)};
    }

    @Child private PrecedenceNode precedenceNode;
    @Child private Length lengthNode;
    @Child private RecursiveLength recursiveLengthNode;

    protected Unlist() {
        this.precedenceNode = PrecedenceNodeFactory.create(null, null);
    }

    @NodeChild(value = "operand")
    protected static abstract class RecursiveLength extends RNode {

        public abstract int executeInt(VirtualFrame frame, Object vector);

        @Child private RecursiveLength recursiveLengthNode;

        private int getRecursiveLength(VirtualFrame frame, Object operand) {
            if (recursiveLengthNode == null) {
                CompilerDirectives.transferToInterpreterAndInvalidate();
                recursiveLengthNode = insert(RecursiveLengthFactory.create(null));
            }
            return recursiveLengthNode.executeInt(frame, operand);
        }

        @Specialization
        @SuppressWarnings("unused")
        public int getLength(RNull vector) {
            return 0;
        }

        @Specialization(order = 10, guards = "!isVectorList")
        public int getLength(RAbstractVector vector) {
            return vector.getLength();
        }

        @Specialization(order = 20, guards = "isVectorList")
        public int getLengthList(VirtualFrame frame, RAbstractVector vector) {
            int totalSize = 0;
            for (int i = 0; i < vector.getLength(); ++i) {
                Object data = vector.getDataAtAsObject(i);
                totalSize += getRecursiveLength(frame, data);
            }
            return totalSize;
        }

        protected boolean isVectorList(RAbstractVector vector) {
            return vector.getElementClass() == Object.class;
        }
    }

    private int getLength(VirtualFrame frame, Object operand) {
        if (lengthNode == null) {
            CompilerDirectives.transferToInterpreterAndInvalidate();
            lengthNode = insert(LengthFactory.create(new RNode[1], getBuiltin(), null));
        }
        return lengthNode.executeInt(frame, operand);
    }

    private int getRecursiveLength(VirtualFrame frame, Object operand) {
        if (recursiveLengthNode == null) {
            CompilerDirectives.transferToInterpreterAndInvalidate();
            recursiveLengthNode = insert(RecursiveLengthFactory.create(null));
        }
        return recursiveLengthNode.executeInt(frame, operand);
    }

    @SuppressWarnings("unused")
    @Specialization
    public RNull unlist(RNull vector, byte recursive, byte useNames) {
        controlVisibility();
        return RNull.instance;
    }

    @SuppressWarnings("unused")
    @Specialization(guards = "!isVectorList")
    public RAbstractVector unlistVector(RAbstractVector vector, byte recursive, byte useNames) {
        controlVisibility();
        return vector;
    }

    @SuppressWarnings("unused")
    @Specialization(guards = "isEmpty")
    public RNull unlistEmptyList(VirtualFrame frame, RList list, byte recursive, byte useNames) {
        controlVisibility();
        return RNull.instance;
    }

    // TODO: initially unlist was on the slow path - hence initial recursive implementation is on
    // the slow path as well; ultimately we may consider (non-recursive) optimization
    @Specialization(guards = "!isEmpty")
    public RAbstractVector unlistList(VirtualFrame frame, RList list, byte recursive, byte useNames) {
        controlVisibility();
        boolean rec = recursive == RRuntime.LOGICAL_TRUE;
        boolean withNames = useNames == RRuntime.LOGICAL_TRUE;

        int precedence = -1;
        int totalSize = 0;
        for (int i = 0; i < list.getLength(); ++i) {
            Object data = list.getDataAt(i);
            precedence = Math.max(precedence, precedenceNode.executeInteger(frame, data, recursive));
            if (recursive == RRuntime.LOGICAL_TRUE) {
                totalSize += getRecursiveLength(frame, data);
            } else {
                totalSize += getLength(frame, data);
            }
        }
        String[] namesData = withNames ? new String[totalSize] : null;
        NamesInfo namesInfo = withNames ? new NamesInfo() : null;
        if (precedence == PrecedenceNode.RAW_PRECEDENCE) {
            byte[] result = new byte[totalSize];
            if (!rec) {
                RStringVector listNames = withNames && list.getNames() != RNull.instance ? (RStringVector) list.getNames() : null;
                int position = 0;
                for (int i = 0; i < list.getLength(); i++) {
                    if (list.getDataAt(i) != RNull.instance) {
                        position = unlistHelperRaw(result, namesData, position, namesInfo, list.getDataAt(i), null, itemName(listNames, i), rec, withNames);
                    }
                }
            } else {
                unlistHelperRaw(result, namesData, 0, namesInfo, list, null, null, rec, withNames);
            }
            return RDataFactory.createRawVector(result, namesInfo != null && namesInfo.namesAssigned ? RDataFactory.createStringVector(namesData, RDataFactory.INCOMPLETE_VECTOR) : null);
        } else if (precedence == PrecedenceNode.LOGICAL_PRECEDENCE) {
            byte[] result = new byte[totalSize];
            if (!rec) {
                RStringVector listNames = withNames && list.getNames() != RNull.instance ? (RStringVector) list.getNames() : null;
                int position = 0;
                for (int i = 0; i < list.getLength(); i++) {
                    if (list.getDataAt(i) != RNull.instance) {
                        position = unlistHelperLogical(result, namesData, position, namesInfo, list.getDataAt(i), null, itemName(listNames, i), rec, withNames);
                    }
                }
            } else {
                unlistHelperLogical(result, namesData, 0, namesInfo, list, null, null, rec, withNames);
            }
            return RDataFactory.createLogicalVector(result, RDataFactory.INCOMPLETE_VECTOR,
                            namesInfo != null && namesInfo.namesAssigned ? RDataFactory.createStringVector(namesData, RDataFactory.INCOMPLETE_VECTOR) : null);
        } else if (precedence == PrecedenceNode.INT_PRECEDENCE) {
            int[] result = new int[totalSize];
            if (!rec) {
                RStringVector listNames = withNames && list.getNames() != RNull.instance ? (RStringVector) list.getNames() : null;
                int position = 0;
                for (int i = 0; i < list.getLength(); i++) {
                    if (list.getDataAt(i) != RNull.instance) {
                        position = unlistHelperInt(result, namesData, position, namesInfo, list.getDataAt(i), null, itemName(listNames, i), rec, withNames);
                    }
                }
            } else {
                unlistHelperInt(result, namesData, 0, namesInfo, list, null, null, rec, withNames);
            }
            return RDataFactory.createIntVector(result, RDataFactory.INCOMPLETE_VECTOR,
                            namesInfo != null && namesInfo.namesAssigned ? RDataFactory.createStringVector(namesData, RDataFactory.INCOMPLETE_VECTOR) : null);
        } else if (precedence == PrecedenceNode.DOUBLE_PRECEDENCE) {
            double[] result = new double[totalSize];
            if (!rec) {
                RStringVector listNames = withNames && list.getNames() != RNull.instance ? (RStringVector) list.getNames() : null;
                int position = 0;
                for (int i = 0; i < list.getLength(); i++) {
                    if (list.getDataAt(i) != RNull.instance) {
                        position = unlistHelperDouble(result, namesData, position, namesInfo, list.getDataAt(i), null, itemName(listNames, i), rec, withNames);
                    }
                }
            } else {
                unlistHelperDouble(result, namesData, 0, namesInfo, list, null, null, rec, withNames);
            }
            return RDataFactory.createDoubleVector(result, RDataFactory.INCOMPLETE_VECTOR,
                            namesInfo != null && namesInfo.namesAssigned ? RDataFactory.createStringVector(namesData, RDataFactory.INCOMPLETE_VECTOR) : null);
        } else if (precedence == PrecedenceNode.COMPLEX_PRECEDENCE) {
            double[] result = new double[totalSize << 1];
            if (!rec) {
                RStringVector listNames = withNames && list.getNames() != RNull.instance ? (RStringVector) list.getNames() : null;
                int position = 0;
                for (int i = 0; i < list.getLength(); i++) {
                    if (list.getDataAt(i) != RNull.instance) {
                        position = unlistHelperComplex(result, namesData, position, namesInfo, list.getDataAt(i), null, itemName(listNames, i), rec, withNames);
                    }
                }
            } else {
                unlistHelperComplex(result, namesData, 0, namesInfo, list, null, null, rec, withNames);
            }
            return RDataFactory.createComplexVector(result, RDataFactory.INCOMPLETE_VECTOR,
                            namesInfo != null && namesInfo.namesAssigned ? RDataFactory.createStringVector(namesData, RDataFactory.INCOMPLETE_VECTOR) : null);
        } else if (precedence == PrecedenceNode.STRING_PRECEDENCE) {
            String[] result = new String[totalSize];
            if (!rec) {
                RStringVector listNames = withNames && list.getNames() != RNull.instance ? (RStringVector) list.getNames() : null;
                int position = 0;
                for (int i = 0; i < list.getLength(); i++) {
                    if (list.getDataAt(i) != RNull.instance) {
                        position = unlistHelperString(result, namesData, position, namesInfo, list.getDataAt(i), null, itemName(listNames, i), rec, withNames);
                    }
                }
            } else {
                unlistHelperString(result, namesData, 0, namesInfo, list, null, null, rec, withNames);
            }
            return RDataFactory.createStringVector(result, RDataFactory.INCOMPLETE_VECTOR,
                            namesInfo != null && namesInfo.namesAssigned ? RDataFactory.createStringVector(namesData, RDataFactory.INCOMPLETE_VECTOR) : null);
        } else if (precedence == PrecedenceNode.LIST_PRECEDENCE) {
            Object[] result = new Object[totalSize];
            if (!rec) {
                RStringVector listNames = withNames && list.getNames() != RNull.instance ? (RStringVector) list.getNames() : null;
                int position = 0;
                for (int i = 0; i < list.getLength(); i++) {
                    if (list.getDataAt(i) != RNull.instance) {
                        position = unlistHelperList(result, namesData, position, namesInfo, list.getDataAt(i), null, itemName(listNames, i), rec, withNames);
                    }
                }
            } else {
                unlistHelperList(result, namesData, 0, namesInfo, list, null, null, rec, withNames);
            }
            return RDataFactory.createList(result, namesInfo != null && namesInfo.namesAssigned ? RDataFactory.createStringVector(namesData, RDataFactory.INCOMPLETE_VECTOR) : null);
        } else {
            throw Utils.nyi();
        }
    }

    protected boolean isVectorList(RAbstractVector vector) {
        return vector.getElementClass() == Object.class;
    }

    private static class NamesInfo {
        public int count = 0;
        public int seqNo = 0;
        public int firstPos = 0;
        public boolean namesAssigned = false;

        public void reset() {
            this.firstPos = -1;
            this.seqNo = 0;
            this.count = 0;
        }
    }

    @SlowPath
    private static int unlistHelperRaw(byte[] result, String[] namesData, int pos, NamesInfo namesInfo, Object o, String outerBase, String tag, boolean recursive, boolean useNames) {
        int position = pos;
        int saveFirstPos = 0;
        int saveSeqNo = 0;
        int saveCount = 0;
        String base = outerBase;
        if (tag != null) {
            base = newBase(outerBase, tag);
            saveFirstPos = namesInfo.firstPos;
            saveSeqNo = namesInfo.seqNo;
            saveCount = namesInfo.count;
            namesInfo.reset();
        }

        if (o instanceof RAbstractVector) {
            RAbstractVector v = (RAbstractVector) o;
            RStringVector listNames = useNames && v.getNames() != RNull.instance ? (RStringVector) v.getNames() : null;
            for (int i = 0; i < v.getLength(); ++i) {
                String name = itemName(listNames, i);
                Object cur = v.getDataAtAsObject(i);
                if (v instanceof RList && recursive) {
                    position = unlistHelperRaw(result, namesData, position, namesInfo, cur, base, name, recursive, useNames);
                } else {
                    assignName(name, base, position, namesData, namesInfo, useNames);
                    result[position++] = unlistValueRaw(cur);
                }
            }
        } else if (o != RNull.instance) {
            assignName(null, base, position, namesData, namesInfo, useNames);
            result[position++] = unlistValueRaw(o);
        }
        fixupName(tag, base, namesData, namesInfo, useNames, saveFirstPos, saveCount, saveSeqNo);
        return position;
    }

    @SlowPath
    private static int unlistHelperLogical(byte[] result, String[] namesData, int pos, NamesInfo namesInfo, Object o, String outerBase, String tag, boolean recursive, boolean useNames) {
        int position = pos;
        int saveFirstPos = 0;
        int saveSeqNo = 0;
        int saveCount = 0;
        String base = outerBase;
        if (tag != null) {
            base = newBase(outerBase, tag);
            saveFirstPos = namesInfo.firstPos;
            saveSeqNo = namesInfo.seqNo;
            saveCount = namesInfo.count;
            namesInfo.reset();
        }

        if (o instanceof RAbstractVector) {
            RAbstractVector v = (RAbstractVector) o;
            RStringVector listNames = useNames && v.getNames() != RNull.instance ? (RStringVector) v.getNames() : null;
            for (int i = 0; i < v.getLength(); ++i) {
                String name = itemName(listNames, i);
                Object cur = v.getDataAtAsObject(i);
                if (v instanceof RList && recursive) {
                    position = unlistHelperLogical(result, namesData, position, namesInfo, cur, base, name, recursive, useNames);
                } else {
                    assignName(name, base, position, namesData, namesInfo, useNames);
                    result[position++] = unlistValueLogical(cur);
                }
            }
        } else if (o != RNull.instance) {
            assignName(null, base, position, namesData, namesInfo, useNames);
            result[position++] = unlistValueLogical(o);
        }
        fixupName(tag, base, namesData, namesInfo, useNames, saveFirstPos, saveCount, saveSeqNo);
        return position;
    }

    @SlowPath
    private static int unlistHelperInt(int[] result, String[] namesData, int pos, NamesInfo namesInfo, Object o, String outerBase, String tag, boolean recursive, boolean useNames) {
        int position = pos;
        int saveFirstPos = 0;
        int saveSeqNo = 0;
        int saveCount = 0;
        String base = outerBase;
        if (tag != null) {
            base = newBase(outerBase, tag);
            saveFirstPos = namesInfo.firstPos;
            saveSeqNo = namesInfo.seqNo;
            saveCount = namesInfo.count;
            namesInfo.reset();
        }

        if (o instanceof RAbstractVector) {
            RAbstractVector v = (RAbstractVector) o;
            RStringVector listNames = useNames && v.getNames() != RNull.instance ? (RStringVector) v.getNames() : null;
            for (int i = 0; i < v.getLength(); ++i) {
                String name = itemName(listNames, i);
                Object cur = v.getDataAtAsObject(i);
                if (v instanceof RList && recursive) {
                    position = unlistHelperInt(result, namesData, position, namesInfo, cur, base, name, recursive, useNames);
                } else {
                    assignName(name, base, position, namesData, namesInfo, useNames);
                    result[position++] = unlistValueInt(cur);
                }
            }
        } else if (o != RNull.instance) {
            assignName(null, base, position, namesData, namesInfo, useNames);
            result[position++] = unlistValueInt(o);
        }
        fixupName(tag, base, namesData, namesInfo, useNames, saveFirstPos, saveCount, saveSeqNo);
        return position;
    }

    @SlowPath
    private static int unlistHelperDouble(double[] result, String[] namesData, int pos, NamesInfo namesInfo, Object o, String outerBase, String tag, boolean recursive, boolean useNames) {
        int position = pos;
        int saveFirstPos = 0;
        int saveSeqNo = 0;
        int saveCount = 0;
        String base = outerBase;
        if (tag != null) {
            base = newBase(outerBase, tag);
            saveFirstPos = namesInfo.firstPos;
            saveSeqNo = namesInfo.seqNo;
            saveCount = namesInfo.count;
            namesInfo.reset();
        }

        if (o instanceof RAbstractVector) {
            RAbstractVector v = (RAbstractVector) o;
            RStringVector listNames = useNames && v.getNames() != RNull.instance ? (RStringVector) v.getNames() : null;
            for (int i = 0; i < v.getLength(); ++i) {
                String name = itemName(listNames, i);
                Object cur = v.getDataAtAsObject(i);
                if (v instanceof RList && recursive) {
                    position = unlistHelperDouble(result, namesData, position, namesInfo, cur, base, name, recursive, useNames);
                } else {
                    assignName(name, base, position, namesData, namesInfo, useNames);
                    result[position++] = unlistValueDouble(cur);
                }
            }
        } else if (o != RNull.instance) {
            assignName(null, base, position, namesData, namesInfo, useNames);
            result[position++] = unlistValueDouble(o);
        }
        fixupName(tag, base, namesData, namesInfo, useNames, saveFirstPos, saveCount, saveSeqNo);
        return position;
    }

    @SlowPath
    private static int unlistHelperComplex(double[] result, String[] namesData, int pos, NamesInfo namesInfo, Object o, String outerBase, String tag, boolean recursive, boolean useNames) {
        int position = pos;
        int saveFirstPos = 0;
        int saveSeqNo = 0;
        int saveCount = 0;
        String base = outerBase;
        if (tag != null) {
            base = newBase(outerBase, tag);
            saveFirstPos = namesInfo.firstPos;
            saveSeqNo = namesInfo.seqNo;
            saveCount = namesInfo.count;
            namesInfo.reset();
        }

        if (o instanceof RAbstractVector) {
            RAbstractVector v = (RAbstractVector) o;
            RStringVector listNames = useNames && v.getNames() != RNull.instance ? (RStringVector) v.getNames() : null;
            for (int i = 0; i < v.getLength(); ++i) {
                String name = itemName(listNames, i);
                Object cur = v.getDataAtAsObject(i);
                if (v instanceof RList && recursive) {
                    position = unlistHelperComplex(result, namesData, position, namesInfo, cur, base, name, recursive, useNames);
                } else {
                    assignName(name, base, position >> 1, namesData, namesInfo, useNames);
                    RComplex val = unlistValueComplex(cur);
                    result[position++] = val.getRealPart();
                    result[position++] = val.getImaginaryPart();
                }
            }
        } else if (o != RNull.instance) {
            assignName(null, base, position >> 1, namesData, namesInfo, useNames);
            RComplex val = unlistValueComplex(o);
            result[position++] = val.getRealPart();
            result[position++] = val.getImaginaryPart();
        }
        fixupName(tag, base, namesData, namesInfo, useNames, saveFirstPos, saveCount, saveSeqNo);
        return position;
    }

    @SlowPath
    private static int unlistHelperList(Object[] result, String[] namesData, int pos, NamesInfo namesInfo, Object o, String outerBase, String tag, boolean recursive, boolean useNames) {
        int position = pos;
        int saveFirstPos = 0;
        int saveSeqNo = 0;
        int saveCount = 0;
        String base = outerBase;
        if (tag != null) {
            base = newBase(outerBase, tag);
            saveFirstPos = namesInfo.firstPos;
            saveSeqNo = namesInfo.seqNo;
            saveCount = namesInfo.count;
            namesInfo.reset();
        }

        if (o instanceof RAbstractVector) {
            RAbstractVector v = (RAbstractVector) o;
            RStringVector listNames = useNames && v.getNames() != RNull.instance ? (RStringVector) v.getNames() : null;
            for (int i = 0; i < v.getLength(); ++i) {
                String name = itemName(listNames, i);
                Object cur = v.getDataAtAsObject(i);
                if (v instanceof RList && recursive) {
                    position = unlistHelperList(result, namesData, position, namesInfo, cur, base, name, recursive, useNames);
                } else {
                    assignName(name, base, position, namesData, namesInfo, useNames);
                    result[position++] = cur;
                }
            }
        } else if (o != RNull.instance) {
            assignName(null, base, position, namesData, namesInfo, useNames);
            result[position++] = o;
        }
        fixupName(tag, base, namesData, namesInfo, useNames, saveFirstPos, saveCount, saveSeqNo);
        return position;
    }

    @SlowPath
    private static int unlistHelperString(String[] result, String[] namesData, int pos, NamesInfo namesInfo, Object o, String outerBase, String tag, boolean recursive, boolean useNames) {
        int position = pos;
        int saveFirstPos = 0;
        int saveSeqNo = 0;
        int saveCount = 0;
        String base = outerBase;
        if (tag != null) {
            base = newBase(outerBase, tag);
            saveFirstPos = namesInfo.firstPos;
            saveSeqNo = namesInfo.seqNo;
            saveCount = namesInfo.count;
            namesInfo.reset();
        }

        if (o instanceof RAbstractVector) {
            RAbstractVector v = (RAbstractVector) o;
            RStringVector listNames = useNames && v.getNames() != RNull.instance ? (RStringVector) v.getNames() : null;
            for (int i = 0; i < v.getLength(); ++i) {
                String name = itemName(listNames, i);
                Object cur = v.getDataAtAsObject(i);
                if (v instanceof RList && recursive) {
                    position = unlistHelperString(result, namesData, position, namesInfo, cur, base, name, recursive, useNames);
                } else {
                    assignName(name, base, position, namesData, namesInfo, useNames);
                    result[position++] = unlistValueString(v.getDataAtAsObject(i));
                }
            }
        } else if (o != RNull.instance) {
            assignName(null, base, position, namesData, namesInfo, useNames);
            result[position++] = unlistValueString(o);
        }
        fixupName(tag, base, namesData, namesInfo, useNames, saveFirstPos, saveCount, saveSeqNo);
        return position;
    }

    private static void fixupName(String tag, String base, String[] namesData, NamesInfo namesInfo, boolean useNames, int saveFirstPos, int saveCount, int saveSeqNo) {
        if (useNames) {
            if (tag != null) {
                if (namesInfo.firstPos >= 0 && namesInfo.count == 1) {
                    namesData[namesInfo.firstPos] = base;
                }
                namesInfo.firstPos = saveFirstPos;
                namesInfo.count = saveCount;
            }
            namesInfo.seqNo = namesInfo.seqNo + saveSeqNo;
        }
    }

    private static void assignName(String name, String base, int position, String[] namesData, NamesInfo namesInfo, boolean useNames) {
        if (useNames) {
            if (name == null && namesInfo.count == 0) {
                namesInfo.firstPos = position;
            }
            namesInfo.count++;
            namesData[position] = newName(base, name, namesInfo);
        }
    }

    private static String itemName(RStringVector names, int i) {
        if (names == null || names.getDataAt(i).equals(RRuntime.NAMES_ATTR_EMPTY_VALUE)) {
            return null;
        } else {
            return names.getDataAt(i);
        }
    }

    private static String newBase(String base, String tag) {
        if (base != null && tag != null) {
            return createCompositeName(base, tag);
        } else if (base != null) {
            return base;
        } else if (tag != null) {
            return tag;
        } else {
            return RRuntime.NAMES_ATTR_EMPTY_VALUE;
        }

    }

    private static String newName(String base, String tag, NamesInfo namesInfo) {
        namesInfo.seqNo++;
        if (base != null && tag != null) {
            namesInfo.namesAssigned = true;
            return createCompositeName(base, tag);
        } else if (base != null) {
            namesInfo.namesAssigned = true;
            return createCompositeName(base, namesInfo.seqNo);
        } else if (tag != null) {
            namesInfo.namesAssigned = true;
            return tag;
        } else {
            return RRuntime.NAMES_ATTR_EMPTY_VALUE;
        }
    }

    @SlowPath
    private static String createCompositeName(String s1, String s2) {
        return s1 + "." + s2;
    }

    @SlowPath
    private static String createCompositeName(String s1, int i) {
        return s1 + i;
    }

    private static String unlistValueString(Object cur) {
        return RRuntime.toString(cur);
    }

    private static RComplex unlistValueComplex(Object dataAtAsObject) {
        if (dataAtAsObject instanceof RComplex) {
            return (RComplex) dataAtAsObject;
        } else if (dataAtAsObject instanceof Double) {
            double result = unlistValueDouble(dataAtAsObject);
            if (RRuntime.isNA(result)) {
                return RRuntime.createComplexNA();
            } else {
                return RDataFactory.createComplex(result, 0.0);
            }
        } else {
            int result = unlistValueInt(dataAtAsObject);
            if (RRuntime.isNA(result)) {
                return RRuntime.createComplexNA();
            } else {
                return RDataFactory.createComplex(result, 0.0);
            }
        }
    }

    private static double unlistValueDouble(Object dataAtAsObject) {
        if (dataAtAsObject instanceof Double) {
            return (double) dataAtAsObject;
        } else {
            int result = unlistValueInt(dataAtAsObject);
            if (RRuntime.isNA(result)) {
                return RRuntime.DOUBLE_NA;
            } else {
                return result;
            }
        }
    }

    private static int unlistValueInt(Object dataAtAsObject) {
        if (dataAtAsObject instanceof RRaw) {
            RRaw rRaw = (RRaw) dataAtAsObject;
            return RRuntime.raw2int(rRaw);
        } else if (dataAtAsObject instanceof Byte) {
            return RRuntime.logical2int((byte) dataAtAsObject);
        } else {
            return (int) dataAtAsObject;
        }
    }

    private static byte unlistValueLogical(Object dataAtAsObject) {
        if (dataAtAsObject instanceof RRaw) {
            RRaw rRaw = (RRaw) dataAtAsObject;
            return RRuntime.raw2logical(rRaw);
        } else {
            return (byte) dataAtAsObject;
        }
    }

    private static byte unlistValueRaw(Object dataAtAsObject) {
        return ((RRaw) dataAtAsObject).getValue();
    }

    public static boolean isEmpty(RAbstractVector vector) {
        return vector.getLength() == 0;
    }
}
