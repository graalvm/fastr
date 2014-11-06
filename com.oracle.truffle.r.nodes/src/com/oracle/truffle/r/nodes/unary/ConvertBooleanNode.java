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
package com.oracle.truffle.r.nodes.unary;

import com.oracle.truffle.api.dsl.*;
import com.oracle.truffle.api.frame.*;
import com.oracle.truffle.api.utilities.*;
import com.oracle.truffle.r.nodes.*;
import com.oracle.truffle.r.runtime.*;
import com.oracle.truffle.r.runtime.data.*;
import com.oracle.truffle.r.runtime.data.model.*;
import com.oracle.truffle.r.runtime.ops.na.*;

public abstract class ConvertBooleanNode extends UnaryNode {

    private final NAProfile naProfile = NAProfile.create();
    private final BranchProfile invalidElementCountBranch = BranchProfile.create();
    private final BranchProfile errorBranch = BranchProfile.create();

    @Override
    public abstract byte executeByte(VirtualFrame frame);

    public abstract byte executeByte(VirtualFrame frame, Object operandValue);

    @Specialization
    protected byte doInt(int value) {
        if (naProfile.isNA(value)) {
            throw RError.error(getEncapsulatingSourceSection(), RError.Message.ARGUMENT_NOT_INTERPRETABLE_LOGICAL);
        }
        return RRuntime.int2logicalNoCheck(value);
    }

    @Specialization
    protected byte doDouble(double value) {
        if (naProfile.isNA(value)) {
            throw RError.error(getEncapsulatingSourceSection(), RError.Message.ARGUMENT_NOT_INTERPRETABLE_LOGICAL);
        }
        return RRuntime.double2logicalNoCheck(value);
    }

    @Specialization
    protected byte doLogical(byte value) {
        if (naProfile.isNA(value)) {
            throw RError.error(getEncapsulatingSourceSection(), RError.Message.NA_UNEXP);
        }
        return value;
    }

    @Specialization
    protected byte doComplex(RComplex value) {
        if (naProfile.isNA(value)) {
            throw RError.error(getEncapsulatingSourceSection(), RError.Message.ARGUMENT_NOT_INTERPRETABLE_LOGICAL);
        }
        return RRuntime.complex2logicalNoCheck(value);
    }

    @Specialization
    protected byte doString(String value) {
        byte logicalValue = RRuntime.string2logicalNoCheck(value);
        if (naProfile.isNA(logicalValue)) {
            throw RError.error(getEncapsulatingSourceSection(), RError.Message.ARGUMENT_NOT_INTERPRETABLE_LOGICAL);
        }
        return logicalValue;
    }

    @Specialization
    protected byte doRaw(RRaw value) {
        return RRuntime.raw2logical(value);
    }

    private void checkLength(RAbstractVector value) {
        if (value.getLength() != 1) {
            invalidElementCountBranch.enter();
            if (value.getLength() == 0) {
                errorBranch.enter();
                throw RError.error(getEncapsulatingSourceSection(), RError.Message.LENGTH_ZERO);
            } else {
                RError.warning(getEncapsulatingSourceSection(), RError.Message.LENGTH_GT_1);
            }
        }
    }

    @Specialization
    protected byte doIntVector(RAbstractIntVector value) {
        checkLength(value);
        return doInt(value.getDataAt(0));
    }

    @Specialization
    protected byte doDoubleVector(RAbstractDoubleVector value) {
        checkLength(value);
        return doDouble(value.getDataAt(0));
    }

    @Specialization
    protected byte doLogicalVector(RLogicalVector value) {
        checkLength(value);
        return doLogical(value.getDataAt(0));
    }

    @Specialization
    protected byte doComplexVector(RComplexVector value) {
        checkLength(value);
        return doComplex(value.getDataAt(0));
    }

    @Specialization
    protected byte doStringVector(RStringVector value) {
        checkLength(value);
        return doString(value.getDataAt(0));
    }

    @Specialization
    protected byte doRawVector(RRawVector value) {
        checkLength(value);
        return doRaw(value.getDataAt(0));
    }

    public static ConvertBooleanNode create(RNode node) {
        if (node instanceof ConvertBooleanNode) {
            return (ConvertBooleanNode) node;
        }
        return ConvertBooleanNodeFactory.create(node);
    }
}
