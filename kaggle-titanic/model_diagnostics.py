#helper functions to do model diagnostics
from sklearn import metrics
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_line, geom_bar, geom_density, geom_hline, ggtitle, xlab, ylab, geom_label, geom_abline, stat_ecdf, theme, element_text, geom_histogram, geom_text

def model_diagnostics(y_test, preds, preds_binary):
    """A function to take outcomes (y_test), fitted values (preds), and binary predictions (preds_binary) 
    from a classifier to generate diagnostic plots for that model. 

    Returns a confusion matrix, precision recall curve, ROC curve, as well as density / distribution plots

    Args:
        y_test (array_like(int)): The true outcomes of the data (int)
        preds (array_like(float)): The fitted values of the classifier (float)
        preds_binary (array_like(int)): The predicted outcomes of the classifier (int)
    """
    dict_scores = {}

    dict_scores['accuracy'] = metrics.accuracy_score(y_test, preds_binary)
    dict_scores['average_precision'] = metrics.average_precision_score(y_true = y_test, y_score = preds)
    dict_scores['f1_score'] = metrics.f1_score(y_test, preds_binary)

    dict_scores

    # Confusion matrix
    cm = metrics.confusion_matrix(y_test, preds_binary)
    cm_display = metrics.ConfusionMatrixDisplay(cm)

    # Precision recall curve
    x = metrics.precision_recall_curve(y_test, preds)

    df_pr_curve = pd.DataFrame()

    df_pr_curve['precision'] = x[0][:-1]
    df_pr_curve['recall'] = x[1][:-1]
    df_pr_curve['threshold'] = x[2]

    auprc = metrics.auc(df_pr_curve['recall'], df_pr_curve['precision'])
    auprc_noskill = np.mean(y_test)
    auprc_text = f"Area under PR curve: {round(auprc, 3)}"

    pr_plot = (
        ggplot(df_pr_curve, aes(x = 'recall', y = 'precision')) 
        + geom_line(size = 1, color = 'orange')
        + geom_hline(aes(yintercept = auprc_noskill), linetype = 'dashed')
        + ggtitle("Precision Recall Curve")
        + geom_label(label = auprc_text, x = np.min(df_pr_curve['recall']), y = np.min(df_pr_curve['precision']), ha = 'left', va = 'bottom', size = 10)
    )

    # Receiver operating characteristic curve
    x = metrics.roc_curve(y_test, preds)
    df_roc_curve = pd.DataFrame()

    df_roc_curve['fpr'] = x[0]
    df_roc_curve['tpr'] = x[1]
    df_roc_curve['threshold'] = x[2]

    auroc = metrics.auc(df_roc_curve['fpr'], df_roc_curve['tpr'])
    auroc_text = f"Area under ROC curve: {round(auroc, 4)}"

    roc_plot = (
        ggplot(df_roc_curve, aes(x = 'fpr', y = 'tpr')) 
        + geom_line(size = 1, color = 'orange')
        + geom_abline(intercept = 0, slope = 1, linetype = 'dashed')
        + xlab("False Positive Rate")
        + ylab("True Positive Rate")
        + ggtitle("Receiver Operating Characteristic Curve")
        + geom_label(label = auroc_text, x = np.max(df_roc_curve['fpr']), y = np.min(df_roc_curve['tpr']), ha = 'right', va = 'bottom', size = 10)
    )

    #ECDF/density plot of predictions
    preds_df = pd.DataFrame(np.vstack([preds, y_test]).T)
    preds_df.columns = ['pred', 'label']

    preds_ecdf_plot = (
        ggplot(preds_df, aes(x = 'pred'))
        + stat_ecdf()
        + ggtitle("Emperical CDF of predictions")
    )

    preds_density_plot = (
        ggplot(preds_df, aes(x = 'pred'))
        + geom_density()
        + ggtitle("Density plot of predictions")
    )

    #Get everything ready to return
    ret_list_scores = [dict_scores, df_pr_curve, df_roc_curve]
    ret_list_plots = [pr_plot, roc_plot, preds_ecdf_plot, preds_density_plot]

    return((ret_list_scores, cm_display, ret_list_plots))


def skf_preds(skf, X, y, estimators):
    """Generate predictions on test folds of StratifiedKFolds, given estimators and original data 

    Args:
        skf (_type_): The StratifiedKFold that the train and test splits come from
        X (numpy array or pandas df): The full training data features
        y (numpy array): The full training data outcomes
        estimators (_type_): The estimators trained on the training data
    """
    preds_list = []
    preds_binary_list = []
    y_list = []

    for i, (_, test_index) in enumerate(skf.split(X, y)):
        preds_list.append(estimators[i].predict_proba(X.loc[test_index, :])[:,1])
        preds_binary_list.append(estimators[i].predict(X.loc[test_index, :]))
        y_list.append(y[test_index])

    return([preds_list, preds_binary_list, y_list])



def model_diagnostics_skf(preds_list, preds_binary_list, y_list):
    """Generate model diagnostics on lists of results from cross validation

    Args:
        preds_list (_type_): _description_
        preds_binary_list (_type_): _description_
        y_list (_type_): _description_
    """

    #base scores (accuracy, average precision, f1 score)
    dict_scores = {
        'accuracy': [],
        'average_precision': [],
        'f1_score': [],
        'auc': [],
        'fold': [],
    }

    for i, (y_test, preds, preds_binary) in enumerate(zip(y_list, preds_list, preds_binary_list)):
        dict_scores['accuracy'].append(metrics.accuracy_score(y_test, preds_binary))
        dict_scores['average_precision'].append(metrics.average_precision_score(y_true = y_test, y_score = preds))
        dict_scores['f1_score'].append(metrics.f1_score(y_test, preds_binary))
        dict_scores['auc'].append(metrics.roc_auc_score(y_test, preds))
        dict_scores['fold'].append(i)


    df_scores = pd.DataFrame(dict_scores)
    df_scores_agg = df_scores.drop('fold', axis = 1).mean()

    # Confusion matrix
    cm_list = []
    for y_test, preds_binary in zip(y_list, preds_binary_list):
        cm_list.append(metrics.confusion_matrix(y_test, preds_binary))

    cm_array = np.stack(cm_list, axis = 0)
    cm_mean = np.mean(cm_array, axis = 0)
    cm_display = metrics.ConfusionMatrixDisplay(cm_mean)

    #precision recall curve

    #initialize variables to store result
    df_pr_curve = pd.DataFrame()
    df_pr_curve_interp = pd.DataFrame()
    base_precision = np.linspace(0, 1, 101)

    for i, (y_test, preds) in enumerate(zip(y_list, preds_list)):
        df_pr_curve_tmp = pd.DataFrame()
        df_pr_curve_interp_tmp = pd.DataFrame()

        precision, recall, threshold = metrics.precision_recall_curve(y_test, preds)
        recall_interp = np.interp(base_precision, precision, recall)

        #populate actual precision and recall for each separate fold's line in the plot
        df_pr_curve_tmp['precision'] = precision[:-1]
        df_pr_curve_tmp['recall'] = recall[:-1]
        df_pr_curve_tmp['threshold'] = threshold
        df_pr_curve_tmp['fold'] = str(i)

        #populate interpolated fpr and tpr for the "mean" line in the plot
        df_pr_curve_interp_tmp['precision'] = base_precision
        df_pr_curve_interp_tmp['recall'] = recall_interp
        df_pr_curve_interp_tmp['fold'] = str(i)
        
        df_pr_curve = pd.concat([df_pr_curve, df_pr_curve_tmp])
        df_pr_curve_interp = pd.concat([df_pr_curve_interp, df_pr_curve_interp_tmp])


    df_pr_mean = (
        df_pr_curve_interp
        .groupby('precision')
        .agg(recall = ('recall', 'mean'))
        .reset_index()
    )

    auprc = metrics.auc(df_pr_mean['precision'], df_pr_mean['recall'])
    auprc_text = f"Area under Precision-Recall curve: {round(auprc, 4)}"
    recall_noskill = np.mean(np.concatenate(y_list))

    pr_plot = (
        ggplot() 
        + geom_line(aes(x = 'precision', y = 'recall'), data = df_pr_mean, size = 1, color = 'orange')
        + geom_line(aes(x = 'precision', y = 'recall', group = 'fold'), data = df_pr_curve, alpha = .07)
        + geom_hline(aes(yintercept = recall_noskill), linetype = 'dashed')
        + ggtitle(f"Precision-Recall Curve \n {auprc_text}")
    )

    #ROC curve
    df_roc_curve = pd.DataFrame()
    df_roc_curve_interp = pd.DataFrame()
    base_fpr = np.linspace(0, 1, 101)

    for i, (y_test, preds) in enumerate(zip(y_list, preds_list)):
        df_roc_curve_tmp = pd.DataFrame()
        df_roc_curve_interp_tmp = pd.DataFrame()

        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0 

        #populate the actual fpr and tpr for the each separate fold's line in the plot
        df_roc_curve_tmp['fpr'] = fpr
        df_roc_curve_tmp['tpr'] = tpr
        df_roc_curve_tmp['threshold'] = threshold
        df_roc_curve_tmp['fold'] = str(i)

        #populate interpolated fpr and tpr for the "mean" line in the plot
        df_roc_curve_interp_tmp['fpr'] = base_fpr
        df_roc_curve_interp_tmp['tpr'] = tpr_interp
        df_roc_curve_interp_tmp['fold'] = str(i)
        
        df_roc_curve = pd.concat([df_roc_curve, df_roc_curve_tmp])
        df_roc_curve_interp = pd.concat([df_roc_curve_interp, df_roc_curve_interp_tmp])


    df_roc_mean = (
        df_roc_curve_interp
        .groupby('fpr')
        .agg(tpr = ('tpr', 'mean'),
            tpr_std = ('tpr', 'std'))
        .reset_index()
        .assign(tpr_upper = lambda x: x.tpr + 1.96 * x.tpr_std,
                tpr_lower = lambda x: x.tpr - 1.96 * x.tpr_std)
    )

    auroc = metrics.auc(df_roc_mean['fpr'], df_roc_mean['tpr'])
    auroc_text = f"Area under ROC curve: {round(auroc, 4)}"

    roc_plot = (
        ggplot() 
        + geom_line(aes(x = 'fpr', y = 'tpr'), data = df_roc_mean, size = 1, color = 'orange')
        + geom_line(aes(x = 'fpr', y = 'tpr', group = 'fold'), data = df_roc_curve, alpha = .05)
        + geom_abline(intercept = 0, slope = 1, linetype = 'dashed')
        + xlab("False Positive Rate")
        + ylab("True Positive Rate")
        + ggtitle(f"Receiver Operating Characteristic Curve \n {auroc_text}")
    )

    ret_list_plots = [pr_plot, roc_plot]
    ret_list = [df_scores, df_scores_agg, cm_display, ret_list_plots]
    return(ret_list)


def summarise_continuous_feature(df_input, column, outcome, n_quantiles = 10):
    
    group_assignments, bins = pd.qcut(df_input[column], q = n_quantiles, retbins = True)
    df_input['quantile'] = group_assignments
    #old line of code cutting data into quantiles
    #pd.cut(df['Age'], bins, labels = np.arange(10))  

    df_mean_outcome = (
        df_input
        .groupby('quantile')
        .agg(
            n = ('quantile', 'count'),
            avg_outcome = (outcome, 'mean')
        )
        .reset_index()
        .round(2)
    )

    df_outcome_counts = (
        df_input
        .astype({outcome: 'string'})
        .groupby(['quantile', outcome])
        .agg(n = (outcome, 'count'))
        .reset_index()
    )

    plot_histogram = (
        ggplot(df_input, aes(x = column))
        + geom_histogram(color = 'orange', fill = 'orange')
    )

    plot_mean_outcome = (
        ggplot(df_mean_outcome, aes(x = 'quantile', y = 'avg_outcome'))
        + geom_bar(stat = 'identity', color = 'green', fill = 'green')
        + xlab(f'Quantile of {column}')
        + ylab(f'Average of {outcome}')
        + theme(axis_text_x=element_text(rotation=-45, hjust=0))
    )

    plot_outcome_counts = (
        ggplot(df_outcome_counts, aes(x = 'quantile', y = 'n', color = outcome, fill = outcome))
        + geom_bar(stat = 'identity', position = 'dodge')
        + xlab(f'Quantile of {column}')
        + theme(axis_text_x=element_text(rotation=-45, hjust=0))
    )

    ret_dict = {
        'mean_outcome_by_quantile': df_mean_outcome,
        'outcome_count_by_quantile': df_outcome_counts,
        'histogram': plot_histogram,
        'plot_mean_outcome_by_quantile': plot_mean_outcome,
        'plot_outcome_counts': plot_outcome_counts,
    }

    return(ret_dict)

#summarise_continuous_feature(df_input = data, column = 'Age', outcome = 'Survived', n_quantiles = 10)