/*
 * Copyright (c) 2018, 2018, Oracle and/or its affiliates. All rights reserved.
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

// Code generated by com.oracle.truffle.r.ffi.codegen.FFITestsCodeGen class run with option '-init'
// All the generated files in testrffi can be regenerated by running 'mx testrfficodegen'
// The following code registers all C functions that wrap RFFI functions and convert SEXP <-> primitive types.
// The definitions of the C functions could be generated by the same Java class (but run without any option)
// RFFI functions that take/return C pointers are ignored
// This code is '#included' into init.c 
CALLDEF(api_Rf_ScalarInteger, 1),
CALLDEF(api_Rf_ScalarLogical, 1),
CALLDEF(api_Rf_ScalarReal, 1),
CALLDEF(api_Rf_ScalarString, 1),
CALLDEF(api_Rf_asInteger, 1),
CALLDEF(api_Rf_asReal, 1),
CALLDEF(api_Rf_asLogical, 1),
CALLDEF(api_Rf_asChar, 1),
CALLDEF(api_Rf_coerceVector, 2),
CALLDEF(api_Rf_mkCharLenCE, 3),
CALLDEF(api_Rf_cons, 2),
CALLDEF(api_Rf_defineVar, 3),
CALLDEF(api_R_getClassDef, 1),
CALLDEF(api_R_do_MAKE_CLASS, 1),
CALLDEF(api_R_do_new_object, 1),
CALLDEF(api_Rf_findVar, 2),
CALLDEF(api_Rf_findVarInFrame, 2),
CALLDEF(api_Rf_findVarInFrame3, 3),
CALLDEF(api_ATTRIB, 1),
CALLDEF(api_Rf_getAttrib, 2),
CALLDEF(api_Rf_setAttrib, 3),
CALLDEF(api_Rf_inherits, 2),
CALLDEF(api_Rf_install, 1),
CALLDEF(api_Rf_installChar, 1),
CALLDEF(api_Rf_lengthgets, 2),
CALLDEF(api_Rf_isString, 1),
CALLDEF(api_Rf_isNull, 1),
CALLDEF(api_Rf_PairToVectorList, 1),
CALLDEF(api_Rf_error, 1),
CALLDEF(api_Rf_warning, 1),
CALLDEF(api_Rf_warningcall, 2),
CALLDEF(api_Rf_errorcall, 2),
CALLDEF(api_Rf_allocVector, 2),
CALLDEF(api_Rf_allocArray, 2),
CALLDEF(api_Rf_allocMatrix, 3),
CALLDEF(api_Rf_nrows, 1),
CALLDEF(api_Rf_ncols, 1),
CALLDEF(api_LENGTH, 1),
CALLDEF(api_SET_STRING_ELT, 3),
CALLDEF(api_SET_VECTOR_ELT, 3),
CALLDEF(api_SET_ATTRIB, 2),
CALLDEF(api_STRING_ELT, 2),
CALLDEF(api_VECTOR_ELT, 2),
CALLDEF(api_NAMED, 1),
CALLDEF(api_SET_OBJECT, 2),
CALLDEF(api_SET_NAMED, 2),
CALLDEF(api_TYPEOF, 1),
CALLDEF(api_Rf_any_duplicated, 2),
CALLDEF(api_Rf_any_duplicated3, 3),
CALLDEF(api_PRINTNAME, 1),
CALLDEF(api_TAG, 1),
CALLDEF(api_CAR, 1),
CALLDEF(api_CAAR, 1),
CALLDEF(api_CDR, 1),
CALLDEF(api_CDAR, 1),
CALLDEF(api_CADR, 1),
CALLDEF(api_CADDR, 1),
CALLDEF(api_CADDDR, 1),
CALLDEF(api_CAD4R, 1),
CALLDEF(api_CDDR, 1),
CALLDEF(api_CDDDR, 1),
CALLDEF(api_SET_TAG, 2),
CALLDEF(api_SETCAR, 2),
CALLDEF(api_SETCDR, 2),
CALLDEF(api_FORMALS, 1),
CALLDEF(api_BODY, 1),
CALLDEF(api_CLOENV, 1),
CALLDEF(api_SET_FORMALS, 2),
CALLDEF(api_SET_BODY, 2),
CALLDEF(api_SET_CLOENV, 2),
CALLDEF(api_SETCADR, 2),
CALLDEF(api_SETCADDR, 2),
CALLDEF(api_SETCADDDR, 2),
CALLDEF(api_SETCAD4R, 2),
CALLDEF(api_SYMVALUE, 1),
CALLDEF(api_SET_SYMVALUE, 2),
CALLDEF(api_R_BindingIsLocked, 2),
CALLDEF(api_R_LockBinding, 2),
CALLDEF(api_R_unLockBinding, 2),
CALLDEF(api_R_FindNamespace, 1),
CALLDEF(api_Rf_eval, 2),
CALLDEF(api_Rf_findFun, 2),
CALLDEF(api_Rf_GetOption1, 1),
CALLDEF(api_Rf_gsetVar, 3),
CALLDEF(api_DUPLICATE_ATTRIB, 2),
CALLDEF(api_R_compute_identical, 3),
CALLDEF(api_Rf_copyListMatrix, 3),
CALLDEF(api_Rf_copyMatrix, 3),
CALLDEF(api_RDEBUG, 1),
CALLDEF(api_SET_RDEBUG, 2),
CALLDEF(api_RSTEP, 1),
CALLDEF(api_SET_RSTEP, 2),
CALLDEF(api_ENCLOS, 1),
CALLDEF(api_PRVALUE, 1),
CALLDEF(api_R_lsInternal3, 3),
CALLDEF(api_R_HomeDir, 0),
CALLDEF(api_IS_S4_OBJECT, 1),
CALLDEF(api_SET_S4_OBJECT, 1),
CALLDEF(api_UNSET_S4_OBJECT, 1),
CALLDEF(api_Rprintf, 1),
CALLDEF(api_GetRNGstate, 0),
CALLDEF(api_PutRNGstate, 0),
CALLDEF(api_unif_rand, 0),
CALLDEF(api_Rf_classgets, 2),
CALLDEF(api_R_ExternalPtrAddr, 1),
CALLDEF(api_R_ExternalPtrTag, 1),
CALLDEF(api_R_ExternalPtrProtected, 1),
CALLDEF(api_R_SetExternalPtrAddr, 2),
CALLDEF(api_R_SetExternalPtrTag, 2),
CALLDEF(api_R_SetExternalPtrProtected, 2),
CALLDEF(api_PRSEEN, 1),
CALLDEF(api_PRENV, 1),
CALLDEF(api_R_PromiseExpr, 1),
CALLDEF(api_PRCODE, 1),
CALLDEF(api_R_new_custom_connection, 4),
CALLDEF(api_R_ReadConnection, 3),
CALLDEF(api_R_WriteConnection, 3),
CALLDEF(api_R_GetConnection, 1),
CALLDEF(api_R_do_slot, 2),
CALLDEF(api_R_do_slot_assign, 3),
CALLDEF(api_Rf_str2type, 1),
CALLDEF(api_Rf_dunif, 4),
CALLDEF(api_Rf_qunif, 5),
CALLDEF(api_Rf_punif, 5),
CALLDEF(api_Rf_runif, 2),
CALLDEF(api_Rf_dchisq, 3),
CALLDEF(api_Rf_pchisq, 4),
CALLDEF(api_Rf_qchisq, 4),
CALLDEF(api_Rf_rchisq, 1),
CALLDEF(api_Rf_dnchisq, 4),
CALLDEF(api_Rf_pnchisq, 5),
CALLDEF(api_Rf_qnchisq, 5),
CALLDEF(api_Rf_rnchisq, 2),
CALLDEF(api_Rf_dnorm4, 4),
CALLDEF(api_Rf_pnorm5, 5),
CALLDEF(api_Rf_qnorm5, 5),
CALLDEF(api_Rf_rnorm, 2),
CALLDEF(api_Rf_dlnorm, 4),
CALLDEF(api_Rf_plnorm, 5),
CALLDEF(api_Rf_qlnorm, 5),
CALLDEF(api_Rf_rlnorm, 2),
CALLDEF(api_Rf_dgamma, 4),
CALLDEF(api_Rf_pgamma, 5),
CALLDEF(api_Rf_qgamma, 5),
CALLDEF(api_Rf_rgamma, 2),
CALLDEF(api_Rf_dbeta, 4),
CALLDEF(api_Rf_pbeta, 5),
CALLDEF(api_Rf_qbeta, 5),
CALLDEF(api_Rf_rbeta, 2),
CALLDEF(api_Rf_df, 4),
CALLDEF(api_Rf_pf, 5),
CALLDEF(api_Rf_qf, 5),
CALLDEF(api_Rf_rf, 2),
CALLDEF(api_Rf_dt, 3),
CALLDEF(api_Rf_pt, 4),
CALLDEF(api_Rf_qt, 4),
CALLDEF(api_Rf_rt, 1),
CALLDEF(api_Rf_dbinom, 4),
CALLDEF(api_Rf_pbinom, 5),
CALLDEF(api_Rf_qbinom, 5),
CALLDEF(api_Rf_rbinom, 2),
CALLDEF(api_Rf_dcauchy, 4),
CALLDEF(api_Rf_pcauchy, 5),
CALLDEF(api_Rf_qcauchy, 5),
CALLDEF(api_Rf_rcauchy, 2),
CALLDEF(api_Rf_dexp, 3),
CALLDEF(api_Rf_pexp, 4),
CALLDEF(api_Rf_qexp, 4),
CALLDEF(api_Rf_rexp, 1),
CALLDEF(api_Rf_dgeom, 3),
CALLDEF(api_Rf_pgeom, 4),
CALLDEF(api_Rf_qgeom, 4),
CALLDEF(api_Rf_rgeom, 1),
CALLDEF(api_Rf_dhyper, 5),
CALLDEF(api_Rf_phyper, 6),
CALLDEF(api_Rf_qhyper, 6),
CALLDEF(api_Rf_rhyper, 3),
CALLDEF(api_Rf_dnbinom, 4),
CALLDEF(api_Rf_pnbinom, 5),
CALLDEF(api_Rf_qnbinom, 5),
CALLDEF(api_Rf_rnbinom, 2),
CALLDEF(api_Rf_dnbinom_mu, 4),
CALLDEF(api_Rf_pnbinom_mu, 5),
CALLDEF(api_Rf_qnbinom_mu, 5),
CALLDEF(api_Rf_rnbinom_mu, 2),
CALLDEF(api_Rf_dpois, 3),
CALLDEF(api_Rf_ppois, 4),
CALLDEF(api_Rf_qpois, 4),
CALLDEF(api_Rf_rpois, 1),
CALLDEF(api_Rf_dweibull, 4),
CALLDEF(api_Rf_pweibull, 5),
CALLDEF(api_Rf_qweibull, 5),
CALLDEF(api_Rf_rweibull, 2),
CALLDEF(api_Rf_dlogis, 4),
CALLDEF(api_Rf_plogis, 5),
CALLDEF(api_Rf_qlogis, 5),
CALLDEF(api_Rf_rlogis, 2),
CALLDEF(api_Rf_dnbeta, 5),
CALLDEF(api_Rf_pnbeta, 6),
CALLDEF(api_Rf_qnbeta, 6),
CALLDEF(api_Rf_dnf, 5),
CALLDEF(api_Rf_pnf, 6),
CALLDEF(api_Rf_qnf, 6),
CALLDEF(api_Rf_dnt, 4),
CALLDEF(api_Rf_pnt, 5),
CALLDEF(api_Rf_qnt, 5),
CALLDEF(api_Rf_ptukey, 6),
CALLDEF(api_Rf_qtukey, 6),
CALLDEF(api_Rf_dwilcox, 4),
CALLDEF(api_Rf_pwilcox, 5),
CALLDEF(api_Rf_qwilcox, 5),
CALLDEF(api_Rf_rwilcox, 2),
CALLDEF(api_Rf_dsignrank, 3),
CALLDEF(api_Rf_psignrank, 4),
CALLDEF(api_Rf_qsignrank, 4),
CALLDEF(api_Rf_rsignrank, 1),
CALLDEF(api_Rf_ftrunc, 1),
CALLDEF(api_Rf_namesgets, 2),
CALLDEF(api_Rf_copyMostAttrib, 2),
CALLDEF(api_Rf_VectorToPairList, 1),
CALLDEF(api_Rf_asCharacterFactor, 1),
CALLDEF(api_Rf_match, 3),
CALLDEF(api_Rf_NonNullStringMatch, 2),
CALLDEF(api_R_has_slot, 2),
CALLDEF(api_Rf_PrintValue, 1),
CALLDEF(api_OBJECT, 1),
// ---- end of generated code