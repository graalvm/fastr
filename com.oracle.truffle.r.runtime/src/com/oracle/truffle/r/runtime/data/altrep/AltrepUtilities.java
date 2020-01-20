package com.oracle.truffle.r.runtime.data.altrep;

import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.Specialization;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.nodes.Node;
import com.oracle.truffle.api.profiles.ConditionProfile;
import com.oracle.truffle.r.runtime.RInternalError;
import com.oracle.truffle.r.runtime.data.RBaseObject;

public class AltrepUtilities {
    public static boolean isAltrep(Object object) {
        return object instanceof RBaseObject && ((RBaseObject) object).isAltRep();
    }

    public static boolean hasCoerceMethodRegistered(Object object) {
        if (!isAltrep(object)) {
            return false;
        }

        if (object instanceof RAltIntegerVec) {
            return ((RAltIntegerVec) object).getDescriptor().isCoerceMethodRegistered();
        } else {
            throw RInternalError.shouldNotReachHere("Unexpected altrep type");
        }
    }

    public static boolean hasDuplicateMethodRegistered(Object object) {
        if (!isAltrep(object)) {
            return false;
        }

        if (object instanceof RAltIntegerVec) {
            return ((RAltIntegerVec) object).getDescriptor().isDuplicateMethodRegistered();
        } else {
            throw RInternalError.shouldNotReachHere("Unexpected altrep type");
        }
    }

    public abstract static class AltrepDuplicateMethodInvoker extends Node {
        public abstract Object execute(Object altrepVector, boolean deep);

        public static AltrepDuplicateMethodInvoker create() {
            return AltrepUtilitiesFactory.AltrepDuplicateMethodInvokerNodeGen.create();
        }

        @Specialization(limit = "3")
        Object doAltInt(RAltIntegerVec altIntVec, boolean deep,
                        @CachedLibrary("altIntVec.getDescriptor().getDuplicateMethod()") InteropLibrary duplicateMethodInterop,
                        @Cached("createBinaryProfile()") ConditionProfile hasMirrorProfile) {
            assert altIntVec.getDescriptor().isDuplicateMethodRegistered();
            return altIntVec.getDescriptor().invokeDuplicateMethodCached(altIntVec, deep, duplicateMethodInterop, hasMirrorProfile);
        }
    }

    public abstract static class AltrepSumMethodInvoker extends Node {
        public abstract Object execute(Object altrepVector, boolean naRm);

        public static AltrepSumMethodInvoker create() {
            return AltrepUtilitiesFactory.AltrepSumMethodInvokerNodeGen.create();
        }

        @Specialization(limit = "3")
        Object doAltInt(RAltIntegerVec altIntVec, boolean naRm,
                        @CachedLibrary("altIntVec.getDescriptor().getSumMethod()") InteropLibrary methodInterop,
                        @Cached("createBinaryProfile()") ConditionProfile hasMirrorProfile) {
            assert altIntVec.getDescriptor().isSumMethodRegistered();
            return altIntVec.getDescriptor().invokeSumMethodCached(altIntVec, naRm, methodInterop, hasMirrorProfile);
        }
    }

    public abstract static class AltrepMaxMethodInvoker extends Node {
        public abstract Object execute(Object altrepVector, boolean naRm);

        public static AltrepMaxMethodInvoker create() {
            return AltrepUtilitiesFactory.AltrepMaxMethodInvokerNodeGen.create();
        }

        @Specialization(limit = "3")
        Object doAltInt(RAltIntegerVec altIntVec, boolean naRm,
                        @CachedLibrary("altIntVec.getDescriptor().getMaxMethod()") InteropLibrary methodInterop,
                        @Cached("createBinaryProfile()") ConditionProfile hasMirrorProfile) {
            assert altIntVec.getDescriptor().isMaxMethodRegistered();
            return altIntVec.getDescriptor().invokeMaxMethodCached(altIntVec, naRm, methodInterop, hasMirrorProfile);
        }
    }

    public abstract static class AltrepMinMethodInvoker extends Node {
        public abstract Object execute(Object altrepVector, boolean naRm);

        public static AltrepMinMethodInvoker create() {
            return AltrepUtilitiesFactory.AltrepMinMethodInvokerNodeGen.create();
        }

        @Specialization(limit = "3")
        Object doAltInt(RAltIntegerVec altIntVec, boolean naRm,
                        @CachedLibrary("altIntVec.getDescriptor().getMinMethod()") InteropLibrary methodInterop,
                        @Cached("createBinaryProfile()") ConditionProfile hasMirrorProfile) {
            assert altIntVec.getDescriptor().isMinMethodRegistered();
            return altIntVec.getDescriptor().invokeMinMethodCached(altIntVec, naRm, methodInterop, hasMirrorProfile);
        }
    }
}