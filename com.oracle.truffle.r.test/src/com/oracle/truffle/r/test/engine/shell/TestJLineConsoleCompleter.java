/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
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
package com.oracle.truffle.r.test.engine.shell;

import com.oracle.truffle.api.vm.PolyglotEngine;
import com.oracle.truffle.r.engine.shell.JLineConsoleCompleter;
import com.oracle.truffle.r.runtime.RCmdOptions;
import com.oracle.truffle.r.runtime.context.ConsoleHandler;
import com.oracle.truffle.r.runtime.context.ContextInfo;
import com.oracle.truffle.r.runtime.context.RContext;
import java.io.File;
import org.junit.Test;

import java.util.LinkedList;
import org.junit.After;
import org.junit.Assert;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import org.junit.Before;

public class TestJLineConsoleCompleter {

    private PolyglotEngine engine;
    private ConsoleHandler consoleHandler;
    private JLineConsoleCompleter consoleCompleter;

    @Before
    public void before() {
        consoleHandler = new DummyConsoleHandler();
        consoleCompleter = new JLineConsoleCompleter(consoleHandler);
        JLineConsoleCompleter.testingMode();
        ContextInfo info = ContextInfo.createNoRestore(RCmdOptions.Client.R, null, RContext.ContextKind.SHARE_NOTHING, null, consoleHandler);
        engine = info.createVM(PolyglotEngine.newBuilder());
    }

    @After
    public void dispose() {
        if (engine != null) {
            engine.dispose();
        }
    }

    @Test
    public void testCompl() {
        assertCompl("", 0);
        assertCompl("", 1);
        assertCompl(" ", 1);

        assertCompl("(", 0);
        assertCompl("(", 1);
        assertCompl("=", 1);
        assertCompl("$", 1);

        assertCompl("strt", 4, "strtoi", "strtrim");
        assertCompl("strto", 5, "strtoi");
        assertCompl("strtoi", 5, "strtoi");
        assertCompl("strtoi", 4, "strtoi", "strtrim");
        assertCompl("strto ", 6);

        assertCompl("base::strt", 10, "base::strtoi", "base::strtrim");
        assertCompl("base:::strt", 11, "base:::strtoi", "base:::strtrim");
        assertCompl("base:::strttrt", 14, "base:::");

        assertCompl("strt(", 4, "strtoi", "strtrim");
        assertCompl("strt(", 5);
        assertCompl("f(strt", 6, "strtoi", "strtrim");
        assertCompl("f(base::strt", 12, "base::strtoi", "base::strtrim");
        assertCompl("f(strt(trt", 6, "strtoi", "strtrim");
        assertCompl("f(strt(trt", 10);
        assertCompl("f(strt(strto", 11, "strtoi", "strtrim");
        assertCompl("f(strt(strto", 12, "strtoi");

        String noName = "_f_f_f_";
        assertCompl(noName + ".", 7);
        assertCompl(noName + ".", 8);
        assertCompl(noName + "." + File.separator, 9);
        assertCompl(noName + "'", 7);
        assertCompl(noName + "'", 8, NOT_EMPTY);
        assertCompl(noName + "'." + File.separator, 8, NOT_EMPTY);
        assertCompl(noName + "'." + File.separator, 9, NOT_EMPTY);
        assertCompl(noName + "'." + File.separator, 10, NOT_EMPTY);
        assertCompl(noName + "\"." + File.separator, 8, NOT_EMPTY);
    }

    // e.g. check if the file path completion returned at least something
    private static final String NOT_EMPTY = "_@_Only.Check.If.Result.Not.Empty_@_";

    private void assertCompl(String buffer, int cursor, String... expected) {
        LinkedList<CharSequence> l = new LinkedList<>();
        consoleCompleter.complete(buffer, cursor, l);

        if (expected == null || expected.length == 0) {
            assertTrue(l.isEmpty());
        } else if (expected.length == 1 && NOT_EMPTY.equals(expected[0])) {
            assertFalse(l.isEmpty());
        } else {
            Assert.assertArrayEquals(expected, l.toArray(new CharSequence[l.size()]));
        }
    }

    private class DummyConsoleHandler implements ConsoleHandler {
        private RContext ctx;

        @Override
        public void println(String s) {
        }

        @Override
        public void print(String s) {
        }

        @Override
        public void printErrorln(String s) {
        }

        @Override
        public void printError(String s) {
        }

        @Override
        public String readLine() {
            return "";
        }

        @Override
        public boolean isInteractive() {
            return false;
        }

        @Override
        public String getPrompt() {
            return "";
        }

        @Override
        public void setPrompt(String prompt) {
        }

        @Override
        public String getInputDescription() {
            return "";
        }

        @Override
        public void setContext(RContext ctx) {
            this.ctx = ctx;
        }

        @Override
        public RContext getContext() {
            return ctx;
        }
    }
}
