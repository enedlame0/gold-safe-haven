*------------------------------------------------------------
* 0.  Load data and build a proper time index
*------------------------------------------------------------
clear
import delimited "C:\Users\Ened Lame\Desktop\business project\data.csv", ///
    varnames(1)  case(lower)

* Excel serial → Stata daily date
replace date = date + date("30dec1899","DMY")
format  date %td

keep if date >= date("01apr2007", "DMY") & date <= date("31mar2009", "DMY") //gfc
//keep if date >= date("01mar2000", "DMY") & date <= date("31oct2002", "DMY") //dot com
//keep if date >= date("01jan2020", "DMY") & date <= date("31dec2020", "DMY") //covid

* A simple trading‑day index avoids irregular‑spacing errors
gen int td_index = _n
tsset td_index

*------------------------------------------------------------
* 1.  Percentage log returns (×100 optional)
*------------------------------------------------------------
gen double ln_xau   = ln(xau)
gen double ln_spx   = ln(spx)
gen double ln_bonds = ln(luattruu)

gen double dr_xau   = (ln_xau   - L.ln_xau)
gen double dr_spx   = (ln_spx   - L.ln_spx)
gen double dr_bonds = (ln_bonds - L.ln_bonds)

drop if missing(dr_xau, dr_spx, dr_bonds)

* ---- 1. Gold vs S&P500 ---------------------------------
mgarch dcc ///
    (dr_xau = , noconstant) ///
    (dr_spx = , noconstant), arch(1) garch(1)

* create all correlations, then inspect the names
predict R*, correlation
describe R*

*===============================================================================
* 2.  NEW: Find and display highest/lowest correlation dates (Corrected)
*===============================================================================

* --- Find the max correlation and its date ---
quietly summarize R_dr_spx_dr_xau
local max_corr = r(max)
quietly summarize date if R_dr_spx_dr_xau == `max_corr' & !missing(date)
local date_max_corr = r(max)

* --- Find the min correlation and its date ---
quietly summarize R_dr_spx_dr_xau
local min_corr = r(min)
quietly summarize date if R_dr_spx_dr_xau == `min_corr' & !missing(date)
local date_min_corr = r(min)

* --- Display the results in the console (Backward-Compatible Version) ---
* The {hline} and as_text/as_result syntax requires Stata 14+.
* This version works on all Stata versions.
display ""
display "----------------------------------------"
display "Correlation Extrema: Gold vs S&P 500"
display "----------------------------------------"
display "Highest correlation: " %4.3f `max_corr' " on " %tdDDmonCCYY `date_max_corr'
display "Lowest correlation:  " %4.3f `min_corr' " on " %tdDDmonCCYY `date_min_corr'
display "----------------------------------------"
display ""

* --- Create a summary table ---
preserve
    keep if date == `date_max_corr' | date == `date_min_corr'
    gen str20 event = "Highest Correlation" if date == `date_max_corr'
    replace event = "Lowest Correlation" if date == `date_min_corr'
    format date %tdDDmonCCYY
    order event date R_dr_spx_dr_xau
    rename R_dr_spx_dr_xau Correlation
    list, separator(0) table clean noobs
restore


*------------------------------------------------------------
* 3.  Plot the dynamic correlation with updated title
*------------------------------------------------------------
summarize date, meanonly
local min_date = r(min)
local max_date = r(max)

* --- Format dates and values for the graph's subtitle ---
local date_max_str : display %tdDDmonYY `date_max_corr'
local date_min_str : display %tdDDmonYY `date_min_corr'
local max_corr_str = strofreal(`max_corr', "%5.3f")
local min_corr_str = strofreal(`min_corr', "%5.3f")

* ==> pick the cross‑correlation (it is R_dr_spx_dr_xau)
twoway line R_dr_spx_dr_xau date, ///
	   title("Dynamic correlation Gold vs S&P 500: COVID-19 Pandemic") ///
       xlabel(`min_date'(365)`max_date', format(%tdCCYY) angle(45)) ///
	   ytitle("ρ̂(t)") xtitle("Year")