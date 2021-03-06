/*
 * Copyright (c) 2020, 2020, Oracle and/or its affiliates. All rights reserved.
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
package com.oracle.truffle.r.runtime.ffi.util;

import com.oracle.truffle.api.dsl.Cached;
import com.oracle.truffle.api.dsl.GenerateUncached;
import com.oracle.truffle.api.dsl.ImportStatic;
import com.oracle.truffle.api.dsl.Specialization;
import com.oracle.truffle.api.interop.InteropLibrary;
import com.oracle.truffle.api.interop.InvalidArrayIndexException;
import com.oracle.truffle.api.interop.UnsupportedMessageException;
import com.oracle.truffle.api.interop.UnsupportedTypeException;
import com.oracle.truffle.api.library.CachedLibrary;
import com.oracle.truffle.api.nodes.Node;
import com.oracle.truffle.api.profiles.ValueProfile;
import com.oracle.truffle.r.runtime.DSLConfig;
import com.oracle.truffle.r.runtime.RInternalError;

/**
 * Encapsulates the operation of writing a value to a target object, which should represent a
 * pointer. This node tries to write the data using array messages of the interop protocol and if
 * that fails it uses the {@code toNative} and {@code asPointer} messages to get the raw pointer and
 * writes to it using unsafe access.
 *
 * Note the {@code index} parameter determines the offset of the write the same way as in
 * {@link NativeMemory#putValue(long, long, Object)}.
 */
@GenerateUncached
@ImportStatic(DSLConfig.class)
public abstract class WritePointerNode extends Node {
    public static WritePointerNode create() {
        return WritePointerNodeGen.create();
    }

    public final void write(Object target, Object value) {
        execute(target, 0, value);
    }

    public abstract void execute(Object target, int index, Object value);

    @Specialization(guards = "targetLib.isArrayElementWritable(target, index)", limit = "getInteropLibraryCacheSize()")
    protected static void putInArray(Object target, int index, Object value,
                    @CachedLibrary("target") InteropLibrary targetLib) {
        try {
            targetLib.writeArrayElement(target, index, value);
        } catch (UnsupportedMessageException | UnsupportedTypeException | InvalidArrayIndexException e) {
            throw RInternalError.shouldNotReachHere();
        }
    }

    @Specialization(guards = {"!targetLib.isArrayElementWritable(target, index)", "targetLib.isPointer(target)"}, limit = "getInteropLibraryCacheSize()")
    protected static void putIntNativeFastPath(Object target, int index, Object value,
                    @Cached("createClassProfile()") ValueProfile valueProfile,
                    @CachedLibrary("target") InteropLibrary targetLib) {
        try {
            long ptr = targetLib.asPointer(target);
            NativeMemory.putValue(ptr, index, valueProfile.profile(value));
        } catch (UnsupportedMessageException e) {
            throw RInternalError.shouldNotReachHere();
        }
    }

    @Specialization(guards = {"!targetLib.isArrayElementWritable(target, index)", "!targetLib.isPointer(target)"}, limit = "getInteropLibraryCacheSize()")
    protected static void putIntNative(Object target, int index, Object value,
                    @Cached("createClassProfile()") ValueProfile valueProfile,
                    @CachedLibrary("target") InteropLibrary targetLib) {
        try {
            targetLib.toNative(target);
            long ptr = targetLib.asPointer(target);
            NativeMemory.putValue(ptr, index, valueProfile.profile(value));
        } catch (UnsupportedMessageException e) {
            throw RInternalError.shouldNotReachHere();
        }
    }
}
