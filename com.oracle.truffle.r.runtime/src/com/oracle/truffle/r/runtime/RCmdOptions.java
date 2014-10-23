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
package com.oracle.truffle.r.runtime;

import java.util.*;

/**
 * (Abstract) definition of the standard R command line options. The setting of the values from the
 * environment is handled in some other class.
 */
public class RCmdOptions {
    public enum Client {
        R {
            @Override
            public String usage() {
                return "\nUsage: R [options] [< infile] [> outfile]\n" + "   or: R CMD command [arguments]\n\n" + "Start R, a system for statistical computation and graphics, with the\n"
                                + "specified options, or invoke an R tool via the 'R CMD' interface.\n";
            }
        },

        RSCRIPT {
            @Override
            public String usage() {
                return "\nUsage: [--options] [-e expr [-e expr2 ...] | file] [args]\n";
            }

        },

        EITHER {
            @Override
            public String usage() {
                Utils.fail("can't call usage() on Client.EITHER");
                return null;
            }
        };

        public abstract String usage();
    }

    private static List<Option<?>> optionList = new ArrayList<>();
    public static final Option<Boolean> HELP = newBooleanOption(true, "h", "help", false, "Print short help message and exit");
    public static final Option<Boolean> VERSION = newBooleanOption(true, "version", false, "Print version info and exit");
    public static final Option<String> ENCODING = newStringOption(false, null, "encoding=ENC", null, "Specify encoding to be used for stdin");
    public static final Option<Boolean> RHOME = newBooleanOption(true, null, "RHOME", false, "Print path to R home directory and exit");
    public static final Option<Boolean> SAVE = newBooleanOption(false, null, "save", false, "Do save workspace at the end of the session");
    public static final Option<Boolean> NO_SAVE = newBooleanOption(true, null, "no-save", false, "Don't save it");
    public static final Option<Boolean> NO_ENVIRON = newBooleanOption(false, null, "no-environ", false, "Don't read the site and user environment files");
    public static final Option<Boolean> NO_SITE_FILE = newBooleanOption(false, null, "no-site-file", false, "Don't read the site-wide Rprofile");
    public static final Option<Boolean> NO_INIT_FILE = newBooleanOption(false, null, "no-init-file", false, "Don't read the user R profile");
    public static final Option<Boolean> RESTORE = newBooleanOption(false, null, "restore", true, "Do restore previously saved objects at startup");
    public static final Option<Boolean> NO_RESTORE_DATA = newBooleanOption(false, null, "no-restore-data", false, "Don't restore previously saved objects");
    public static final Option<Boolean> NO_RESTORE_HISTORY = newBooleanOption(false, null, "no-restore-history", false, "Don't restore the R history file");
    public static final Option<Boolean> NO_RESTORE = newBooleanOption(false, null, "no-restore", false, "Don't restore anything");
    public static final Option<Boolean> VANILLA = newBooleanOption(true, null, "vanilla", false, "Combine --no-save, --no-restore, --no-site-file,\n"
                    + "                          --no-init-file and --no-environ");
    public static final Option<Boolean> NO_READLINE = newBooleanOption(false, null, "no-readline", false, "Don't use readline for command-line editing");
    public static final Option<String> MAX_PPSIZE = newStringOption(false, null, "max-ppsize", null, "Set max size of protect stack to N");
    public static final Option<Boolean> QUIET = newBooleanOption(true, "q", "quiet", false, "Don't print startup message");
    public static final Option<Boolean> SILENT = newBooleanOption(true, "silent", false, "Same as --quiet");
    public static final Option<Boolean> SLAVE = newBooleanOption(true, "slave", false, "Make R run as quietly as possible");
    public static final Option<Boolean> INTERACTIVE = newBooleanOption(false, "interactive", false, "Force an interactive session");
    public static final Option<Boolean> VERBOSE = newBooleanOption(false, "verbose", false, "Print more information about progress");
    public static final Option<String> DEBUGGER = newStringOption(true, "d", "debugger=NAME", null, "Run R through debugger NAME");
    public static final Option<String> DEBUGGER_ARGS = newStringOption(false, null, "debugger-args=ARGS", null, "Pass ARGS as arguments to the debugger");
    public static final Option<String> GUI = newStringOption(false, "g TYPE", "gui=TYPE", null, "Use TYPE as GUI; possible values are 'X11' (default)\n" + "                          and 'Tk'.");
    public static final Option<String> ARCH = newStringOption(false, null, "arch=NAME", null, "Specify a sub-architecture");
    public static final Option<Boolean> ARGS = newBooleanOption(true, "args", false, "Skip the rest of the command line");
    public static final Option<String> FILE = newStringOption(true, "f FILE", "file=FILE", null, "Take input from 'FILE'");
    public static final Option<List<String>> EXPR = newStringListOption(true, "e EXPR", null, null, "Execute 'EXPR' and exit");
    public static final Option<String> DEFAULT_PACKAGES = newStringOption(Client.RSCRIPT, false, null, "default-packages=list", null, "Where 'list' is a comma-separated set\n"
                    + "                          of package names, or 'NULL'");

    public static Option<Boolean> newBooleanOption(boolean implemented, String name, boolean defaultValue, String help) {
        return newBooleanOption(implemented, null, name, defaultValue, help);
    }

    public static Option<Boolean> newBooleanOption(boolean implemented, String shortName, String name, boolean defaultValue, String help) {
        Option<Boolean> option = new Option<>(implemented, OptionType.BOOLEAN, shortName, name, help, defaultValue);
        optionList.add(option);
        return option;
    }

    public static Option<String> newStringOption(boolean implemented, String shortName, String name, String defaultValue, String help) {
        return newStringOption(Client.EITHER, implemented, shortName, name, defaultValue, help);
    }

    public static Option<String> newStringOption(Client client, boolean implemented, String shortName, String name, String defaultValue, String help) {
        Option<String> option = new Option<>(client, implemented, OptionType.STRING, shortName, name, help, defaultValue);
        optionList.add(option);
        return option;
    }

    public static Option<List<String>> newStringListOption(boolean implemented, String shortName, String name, List<String> defaultValue, String help) {
        Option<List<String>> option = new Option<>(Client.EITHER, implemented, OptionType.REPEATED_STRING, shortName, name, help, defaultValue);
        optionList.add(option);
        return option;
    }

    public static List<Option<?>> optionList() {
        return optionList;
    }

    public static enum OptionType {
        BOOLEAN,
        STRING,
        REPEATED_STRING
    }

    public static class Option<T> {
        public final OptionType type;
        /**
         * The option name prefixed by {@code --} or {@code null} if no {@code --} form.
         */
        private final String name;
        /**
         * The short option name prefixed by {@code -} or {@code null} if no {@code -} form.
         */
        private final String shortName;
        /**
         * The '=' separated suffix, e.g. {@code --file=FILE}.
         */
        private String suffix;
        /**
         * The space separated suffix, e.g. {@code -g TYPE}.
         */
        private String shortSuffix;
        /**
         * The help text.
         */
        public final String help;
        /**
         * The value, either from the default or set on the command line.
         */
        private T value;
        /**
         * Set {@code true} iff the short form of a {@code OptionType.STRING} option matched.
         */
        private boolean matchedShort;
        /**
         * Temporary field indicating not implemented.
         */
        public final boolean implemented;

        /**
         * Option specificity.
         */
        public final Client client;

        Option(Client client, boolean implemented, OptionType type, String shortName, String name, String help, T defaultValue) {
            this.client = client;
            this.implemented = implemented;
            this.type = type;
            this.shortName = shortName == null ? null : shortKey(shortName);
            this.name = name == null ? null : key(name);
            this.help = help;
            this.value = defaultValue;
        }

        Option(boolean implemented, OptionType type, String shortName, String name, String help, T defaultValue) {
            this(Client.EITHER, implemented, type, shortName, name, help, defaultValue);
        }

        private static boolean noPrefix(String arg) {
            return arg.equals("RHOME") || arg.equals("CMD");
        }

        private String key(String keyName) {
            if (noPrefix(keyName)) {
                return keyName;
            }
            String xName = keyName;
            int eqx = keyName.indexOf('=');
            if (eqx > 0) {
                xName = keyName.substring(0, eqx);
                suffix = keyName.substring(eqx);
            }
            return "--" + xName;
        }

        private String shortKey(String keyName) {
            String xName = keyName;
            int spx = keyName.indexOf(' ');
            if (spx > 0) {
                xName = keyName.substring(0, spx);
                shortSuffix = keyName.substring(spx);
            }
            return "-" + xName;
        }

        public T getValue() {
            return value;
        }

        boolean matches(String arg) {
            if (shortName != null && arg.equals(shortName)) {
                return true;
            }
            if (name != null && arg.equals(name)) {
                return true;
            }
            return false;
        }

        @SuppressWarnings("unchecked")
        public void setValue(boolean value) {
            this.value = (T) new Boolean(value);
        }

        @SuppressWarnings("unchecked")
        public void setValue(String value) {
            if (type == OptionType.REPEATED_STRING) {
                if (this.value == null) {
                    this.value = (T) new ArrayList<String>();
                }
                ((List<String>) this.value).add(value);
            } else {
                this.value = (T) value;
            }
        }

        public String getHelpName() {
            String result = "";
            if (shortName != null) {
                result = shortName;
                if (shortSuffix != null) {
                    result += shortSuffix;
                }
                if (name != null) {
                    result += ", ";
                }
            }
            if (name != null) {
                result = result + name;
            }
            if (suffix != null) {
                result += suffix;
            }
            return result;
        }

        public boolean matchedShort() {
            return matchedShort;
        }
    }

    public static Option<?> matchOption(String arg) {
        for (Option<?> option : optionList) {
            if (option.type == OptionType.BOOLEAN) {
                // these must match exactly
                if (option.matches(arg)) {
                    return option;
                }
            } else if (option.type == OptionType.STRING || option.type == OptionType.REPEATED_STRING) {
                // short forms must match exactly (and consume next argument)
                if (option.shortName != null && option.shortName.equals(arg)) {
                    option.matchedShort = true;
                    return option;
                } else if (arg.indexOf('=') > 0 && option.name != null && arg.startsWith(option.name)) {
                    return option;
                }
            }
        }
        return null;
    }

}
