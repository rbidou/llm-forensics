/* LLM Forensics - Reports (Tools): the reports saved by the individual tools. */

document.addEventListener('DOMContentLoaded', () => {
    initReportsPage((report) => report.analysis !== ASSESSMENT_ANALYSIS, toolsReportRow);
});
