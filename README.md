# SWMM Comparison App (Enhanced Fork – Florida CRS Support)

This is a **fork** of the original SWMM Comparison App by **@meyerd851-lab**, with enhancements focused on **local coordinate system support for Florida (EPSG:2882)** and improved projection handling for SWMM models that use State Plane coordinates in US survey feet.

**Original Project:**
- Repository: https://github.com/meyerd851-lab/SWMM_Comparison

---

## 🔄 What’s New in This Fork

| Feature | Status | Notes |
|--------|:------:|------|
| Added **EPSG:2882** (NAD83(HARN) / Florida West US-ft) projection support | ✅ | Correct proj4 parameters + UI dropdown update |
| Set **Florida CRS as default** | ✅ | Matching typical SWMM geometry for Tampa Bay area |
| Projection dropdown now syncs UI and internal CRS state | ✅ | Prevents UI/map mismatch issues |
| Code cleaned + ready for additional regional CRS expansion | ✅ | East/North/2011 variants can be added easily |

---

- Live App: https://mgools12.github.io/SWMM_inp_Comparison
- Repository: https://github.com/mgools12/SWMM_inp_Comparison

A browser-based tool for comparing EPA SWMM .inp files. It identifies and visualizes differences between two model versions directly in the browser—no installation or backend required.

**Overview**

The SWMM Comparison App highlights added, removed, and changed elements between two models using an interactive table and map interface. It supports geometry, hydraulics, subcatchments, infiltration parameters, and hydrographs, providing both a summary and detailed view of changes.

**Features**

INP File Comparison: Compares two .inp files across all SWMM sections.

Interactive Map: Visualizes added, removed, changed, and unchanged elements with Leaflet basemaps.

Detailed Table View: Shows line-by-line changes with side-by-side popups for each element.

Hydrograph Support: Compares RTK parameters by month and duration bin with delta calculations.

Session Save/Load: Exports or restores comparison sessions via .json files.

Excel Export: Generates a formatted .xlsx summary of all detected differences.

Offline Use: Runs entirely client-side in any modern browser.

**Technologies**

Frontend: HTML, CSS, JavaScript

Mapping: Leaflet.js, proj4.js

Processing: Web Workers

Export: SheetJS (xlsx)

Hosting: GitHub Pages

Pyodide: Allows Python code to execute natively in the browser, letting the app perform complex model comparisons locally on your device without any external processing.
