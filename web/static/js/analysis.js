/* LLM Forensics - Analysis: the reports saved by the model-analysis workflow. */

document.addEventListener('DOMContentLoaded', () => {
    initReportsPage((report) => report.analysis === ASSESSMENT_ANALYSIS, analysisReportRow);
});
