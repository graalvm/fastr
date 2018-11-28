/*
 * Copyright (c) 2018, Oracle and/or its affiliates. All rights reserved.
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
package com.oracle.truffle.r.nodes.builtin.base.infix.special;

import com.oracle.truffle.api.CompilerDirectives;
import com.oracle.truffle.api.dsl.Fallback;
import com.oracle.truffle.api.dsl.NodeChild;
import com.oracle.truffle.api.dsl.Specialization;
import com.oracle.truffle.api.nodes.NodeCost;
import com.oracle.truffle.api.nodes.NodeInfo;
import com.oracle.truffle.r.nodes.helpers.AccessListField;
import com.oracle.truffle.r.runtime.builtins.RSpecialFactory;
import com.oracle.truffle.r.runtime.data.RList;
import com.oracle.truffle.r.runtime.nodes.RNode;

@NodeInfo(cost = NodeCost.NONE)
@NodeChild(value = "arguments", type = RNode[].class)
public abstract class AccessFieldSpecial extends RNode {

    @Child private AccessListField accessListField;

    @Specialization
    public Object doList(RList list, String field) {
        // Note: special call construction turns lookups into string constants for field accesses
        if (accessListField == null) {
            CompilerDirectives.transferToInterpreterAndInvalidate();
            accessListField = insert(AccessListField.create());
        }
        Object result = accessListField.execute(list, field);
        if (result == null) {
            throw RSpecialFactory.throwFullCallNeeded();
        } else {
            return result;
        }
    }

    @Fallback
    @SuppressWarnings("unused")
    public Object doFallback(Object container, Object field) {
        throw RSpecialFactory.throwFullCallNeeded();
    }
}
