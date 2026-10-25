slides
├── 00_intro_machine_learning.qmd
├── 01_linear_regression.qmd
├── 02_logistic_regression.qmd
├── 03_overfitting_regularization.qmd
├── 04_support_vector_machines.qmd
├── 05_pytorch.qmd
├── 06_appendix_matrix_calculus.qmd
├── 06_neural_networks.qmd
├── 07_convolutional_neural_networks.qmd
├── 08_recurrent_neural_networks.qmd
├── beamercolorthemeCaltechBeaver.sty
├── beamerthemeCaltech.sty
├── data
│   ├── diabetes_bmi.csv
│   ├── diabetes.csv
│   ├── dinosaurs.txt
│   └── MNIST
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
├── figures
│   ├── backprop-math-sketch
│   │   ├── chain_deep.png
│   │   ├── chain_step1.png
│   │   ├── chain_step2.png
│   │   ├── chain_step3.png
│   │   ├── gradcheck.png
│   │   └── jacobians.png
│   ├── backprop-sketch
│   │   ├── backprop_flow.png
│   │   ├── blocks.png
│   │   ├── deep_network.png
│   │   ├── digits_predictions.png
│   │   ├── digits_training.png
│   │   ├── logreg_graph.png
│   │   ├── simple_graph_step1.png
│   │   ├── simple_graph_step2.png
│   │   └── vanishing.png
│   ├── cnn-sketch
│   │   ├── augment.png
│   │   ├── cnn_vs_mlp.png
│   │   ├── conv_step1.png
│   │   ├── conv_step2.png
│   │   ├── conv_step3.png
│   │   ├── cv_tasks.png
│   │   ├── edge_example.png
│   │   ├── example_net.png
│   │   ├── filters.png
│   │   ├── learned_filters.png
│   │   ├── lenet5.png
│   │   ├── nst_example_1.png
│   │   ├── nst_example_2.png
│   │   ├── nst_iterations.png
│   │   ├── one_shot.png
│   │   ├── padding.png
│   │   ├── pooling.png
│   │   ├── siamese.png
│   │   ├── sobel.png
│   │   ├── stride_step1.png
│   │   ├── stride_step2.png
│   │   ├── stride_step3.png
│   │   ├── transfer.png
│   │   ├── triplet.png
│   │   ├── volume_one_filter.png
│   │   └── volume_two_filters.png
│   ├── intro-sketch
│   │   ├── house_step1.png
│   │   ├── house_step2.png
│   │   ├── house_step3.png
│   │   ├── ml_types.png
│   │   ├── regression_vs_classification.png
│   │   ├── spot_the_bug.png
│   │   ├── supervised_diagram.png
│   │   ├── vegetables_step1.png
│   │   ├── vegetables_step2.png
│   │   └── venn.png
│   ├── linear-regression-sketch
│   │   ├── abs_vs_squared.png
│   │   ├── bmi_data_step1.png
│   │   ├── bmi_data_step2.png
│   │   ├── bowl_surface_contour.png
│   │   ├── candidate_lines.png
│   │   ├── gd_in_action.png
│   │   ├── gd_landscape.png
│   │   ├── gd_step_negative_slope.png
│   │   ├── gd_step_positive_slope.png
│   │   ├── gradient_direction_step1.png
│   │   ├── gradient_direction_step2.png
│   │   ├── lr_diverge.png
│   │   ├── lr_large.png
│   │   ├── lr_regimes.png
│   │   ├── lr_small.png
│   │   ├── pipeline.png
│   │   ├── residuals_many_step1.png
│   │   ├── residuals_many_step2.png
│   │   ├── residuals_two_points_step1.png
│   │   ├── residuals_two_points_step2.png
│   │   ├── residuals_two_points_step3.png
│   │   ├── slope_intercept.png
│   │   ├── stationary_1d_step1.png
│   │   ├── stationary_1d_step2.png
│   │   ├── stationary_2d.png
│   │   ├── toy_cost_step1.png
│   │   ├── toy_cost_step2.png
│   │   ├── toy_cost_step3.png
│   │   ├── toy_cost_step4.png
│   │   ├── toy_cost_surface_contour.png
│   │   └── train_test.png
│   ├── logistic-regression-binary-sketch
│   │   ├── boundary_ellipse_step1.png
│   │   ├── boundary_ellipse_step2.png
│   │   ├── boundary_linear_step1.png
│   │   ├── boundary_linear_step2.png
│   │   ├── boundary_wiggly.png
│   │   ├── cost_surfaces.png
│   │   ├── gd_training.png
│   │   ├── linreg_fail_step1.png
│   │   ├── linreg_fail_step2.png
│   │   ├── linreg_fail_step3.png
│   │   ├── linreg_fail_step4.png
│   │   ├── logistic_fit_step1.png
│   │   ├── logistic_fit_step2.png
│   │   ├── loss_curves.png
│   │   ├── model_diagram.png
│   │   ├── sigmoid.png
│   │   ├── sigmoid_threshold.png
│   │   ├── sklearn_breast_cancer.png
│   │   ├── threshold_radius_step1.png
│   │   └── threshold_radius_step2.png
│   ├── logistic-regression-multiclass-sketch
│   │   ├── binary_vs_multiclass.png
│   │   ├── iris_confusion.png
│   │   ├── iris_regions.png
│   │   ├── one_vs_rest.png
│   │   ├── probabilities.png
│   │   ├── softmax_example.png
│   │   ├── toy_step1.png
│   │   └── toy_step2.png
│   ├── multi-linear-regression-sketch
│   │   ├── bad_learning_curves.png
│   │   ├── feature_engineering_area.png
│   │   ├── feature_engineering_pipeline.png
│   │   ├── learning_curve.png
│   │   ├── lr_sweep.png
│   │   ├── normal_equation_example.png
│   │   ├── plane_fit.png
│   │   ├── poly_step1.png
│   │   ├── poly_step2.png
│   │   ├── poly_step3.png
│   │   ├── poly_step4.png
│   │   ├── r2_illustration.png
│   │   ├── scaling_comparison.png
│   │   ├── scaling_contours.png
│   │   ├── scaling_learning_curves.png
│   │   ├── scaling_mean.png
│   │   ├── scaling_minmax.png
│   │   ├── scaling_raw.png
│   │   ├── scaling_zscore.png
│   │   ├── sklearn_1d_example.png
│   │   └── vectorization.png
│   ├── neural-network-sketch
│   │   ├── activations.png
│   │   ├── architectures_step1.png
│   │   ├── architectures_step2.png
│   │   ├── architectures_step3.png
│   │   ├── housing_network_step1.png
│   │   ├── housing_network_step2.png
│   │   ├── housing_network_step3.png
│   │   ├── housing_relu.png
│   │   ├── image_to_vector.png
│   │   ├── mnist_digits.png
│   │   ├── neuron_zoom.png
│   │   ├── two_layer_step1.png
│   │   ├── two_layer_step2.png
│   │   ├── two_layer_step3.png
│   │   ├── why_nonlinear.png
│   │   └── zero_vs_one.png
│   ├── overfitting-sketch
│   │   ├── bias_variance.png
│   │   ├── data.png
│   │   ├── fit_classification_step1.png
│   │   ├── fit_classification_step2.png
│   │   ├── fit_classification_step3.png
│   │   ├── fit_regression_step1.png
│   │   ├── fit_regression_step2.png
│   │   ├── fit_regression_step3.png
│   │   ├── lambda_sweep.png
│   │   ├── more_data.png
│   │   ├── regularised_classification.png
│   │   ├── regularised_fit.png
│   │   ├── regularised_weights.png
│   │   ├── validation_degree.png
│   │   └── validation_lambda.png
│   ├── pytorch-sketch
│   │   ├── course_notebooks.png
│   │   ├── graph_step1.png
│   │   ├── graph_step2.png
│   │   ├── hpc_workflow.png
│   │   ├── learning_curve.png
│   │   ├── minibatch_curves.png
│   │   ├── tensor_ranks.png
│   │   └── training_loop.png
│   ├── rnn-sketch
│   │   ├── bidirectional.png
│   │   ├── bptt.png
│   │   ├── deep_rnn.png
│   │   ├── dense_on_sequence.png
│   │   ├── dino_loss.png
│   │   ├── embedding_map.png
│   │   ├── gradient_lstm.png
│   │   ├── gradient_norms.png
│   │   ├── gru.png
│   │   ├── lm_sampling.png
│   │   ├── lm_training.png
│   │   ├── lstm.png
│   │   ├── one_hot.png
│   │   ├── rnn_cell.png
│   │   ├── rnn_types.png
│   │   ├── unrolled_step1.png
│   │   └── unrolled_step2.png
│   ├── svm-sketch
│   │   ├── geometry.png
│   │   ├── hinge_vs_logistic.png
│   │   ├── lagrange_equality.png
│   │   ├── lagrange_inequality.png
│   │   ├── rbf_grid.png
│   │   ├── soft_alphas.png
│   │   ├── svm_kernel.png
│   │   ├── svm_margin_step1.png
│   │   ├── svm_margin_step2.png
│   │   ├── svm_soft_margin.png
│   │   └── worked_example.png
│   ├── UoG_colour.eps
│   ├── UoG_colour-eps-converted-to.pdf
│   └── UoG_colour.png
├── filters
│   ├── build.lua
│   └── theme.lua
├── output_slides
│   ├── 00_intro_machine_learning.html
│   ├── 00_intro_machine_learning.pdf
│   ├── 01_linear_regression.html
│   ├── 01_linear_regression.pdf
│   ├── 02_logistic_regression.html
│   ├── 02_logistic_regression.pdf
│   ├── 03_overfitting_regularization.html
│   ├── 03_overfitting_regularization.pdf
│   ├── 04_support_vector_machines.html
│   ├── 04_support_vector_machines.pdf
│   ├── 05_pytorch.html
│   ├── 05_pytorch.pdf
│   ├── 06_appendix_matrix_calculus.html
│   ├── 06_appendix_matrix_calculus.pdf
│   ├── 06_neural_networks.html
│   ├── 06_neural_networks.pdf
│   ├── 07_convolutional_neural_networks.html
│   ├── 07_convolutional_neural_networks.pdf
│   ├── 08_recurrent_neural_networks.html
│   ├── 08_recurrent_neural_networks.pdf
│   └── figures
│       ├── backprop-math-sketch
│       │   ├── chain_deep.png
│       │   ├── chain_step1.png
│       │   ├── chain_step2.png
│       │   ├── chain_step3.png
│       │   ├── gradcheck.png
│       │   └── jacobians.png
│       ├── backprop-sketch
│       │   ├── backprop_flow.png
│       │   ├── blocks.png
│       │   ├── deep_network.png
│       │   ├── digits_predictions.png
│       │   ├── digits_training.png
│       │   ├── logreg_graph.png
│       │   ├── simple_graph_step1.png
│       │   ├── simple_graph_step2.png
│       │   └── vanishing.png
│       ├── cnn-sketch
│       │   ├── augment.png
│       │   ├── cnn_vs_mlp.png
│       │   ├── conv_step1.png
│       │   ├── conv_step2.png
│       │   ├── conv_step3.png
│       │   ├── cv_tasks.png
│       │   ├── edge_example.png
│       │   ├── example_net.png
│       │   ├── filters.png
│       │   ├── learned_filters.png
│       │   ├── lenet5.png
│       │   ├── nst_example_1.png
│       │   ├── nst_example_2.png
│       │   ├── nst_iterations.png
│       │   ├── one_shot.png
│       │   ├── padding.png
│       │   ├── pooling.png
│       │   ├── siamese.png
│       │   ├── sobel.png
│       │   ├── stride_step1.png
│       │   ├── stride_step2.png
│       │   ├── stride_step3.png
│       │   ├── transfer.png
│       │   ├── triplet.png
│       │   ├── volume_one_filter.png
│       │   └── volume_two_filters.png
│       ├── intro-sketch
│       │   ├── house_step1.png
│       │   ├── house_step2.png
│       │   ├── house_step3.png
│       │   ├── ml_types.png
│       │   ├── regression_vs_classification.png
│       │   ├── spot_the_bug.png
│       │   ├── supervised_diagram.png
│       │   ├── vegetables_step1.png
│       │   ├── vegetables_step2.png
│       │   └── venn.png
│       ├── linear-regression-sketch
│       │   ├── abs_vs_squared.png
│       │   ├── bmi_data_step1.png
│       │   ├── bmi_data_step2.png
│       │   ├── bowl_surface_contour.png
│       │   ├── candidate_lines.png
│       │   ├── gd_in_action.png
│       │   ├── gd_landscape.png
│       │   ├── gd_step_negative_slope.png
│       │   ├── gd_step_positive_slope.png
│       │   ├── gradient_direction_step1.png
│       │   ├── gradient_direction_step2.png
│       │   ├── lr_diverge.png
│       │   ├── lr_large.png
│       │   ├── lr_regimes.png
│       │   ├── lr_small.png
│       │   ├── pipeline.png
│       │   ├── residuals_many_step1.png
│       │   ├── residuals_many_step2.png
│       │   ├── residuals_two_points_step1.png
│       │   ├── residuals_two_points_step2.png
│       │   ├── residuals_two_points_step3.png
│       │   ├── slope_intercept.png
│       │   ├── stationary_1d_step1.png
│       │   ├── stationary_1d_step2.png
│       │   ├── stationary_2d.png
│       │   ├── toy_cost_step1.png
│       │   ├── toy_cost_step2.png
│       │   ├── toy_cost_step3.png
│       │   ├── toy_cost_step4.png
│       │   ├── toy_cost_surface_contour.png
│       │   └── train_test.png
│       ├── logistic-regression-binary-sketch
│       │   ├── boundary_ellipse_step1.png
│       │   ├── boundary_ellipse_step2.png
│       │   ├── boundary_linear_step1.png
│       │   ├── boundary_linear_step2.png
│       │   ├── boundary_wiggly.png
│       │   ├── cost_surfaces.png
│       │   ├── gd_training.png
│       │   ├── linreg_fail_step1.png
│       │   ├── linreg_fail_step2.png
│       │   ├── linreg_fail_step3.png
│       │   ├── linreg_fail_step4.png
│       │   ├── logistic_fit_step1.png
│       │   ├── logistic_fit_step2.png
│       │   ├── loss_curves.png
│       │   ├── model_diagram.png
│       │   ├── sigmoid.png
│       │   ├── sigmoid_threshold.png
│       │   ├── sklearn_breast_cancer.png
│       │   ├── threshold_radius_step1.png
│       │   └── threshold_radius_step2.png
│       ├── logistic-regression-multiclass-sketch
│       │   ├── binary_vs_multiclass.png
│       │   ├── iris_confusion.png
│       │   ├── iris_regions.png
│       │   ├── one_vs_rest.png
│       │   ├── probabilities.png
│       │   ├── softmax_example.png
│       │   ├── toy_step1.png
│       │   └── toy_step2.png
│       ├── multi-linear-regression-sketch
│       │   ├── bad_learning_curves.png
│       │   ├── feature_engineering_area.png
│       │   ├── feature_engineering_pipeline.png
│       │   ├── learning_curve.png
│       │   ├── lr_sweep.png
│       │   ├── normal_equation_example.png
│       │   ├── plane_fit.png
│       │   ├── poly_step1.png
│       │   ├── poly_step2.png
│       │   ├── poly_step3.png
│       │   ├── poly_step4.png
│       │   ├── r2_illustration.png
│       │   ├── scaling_comparison.png
│       │   ├── scaling_contours.png
│       │   ├── scaling_learning_curves.png
│       │   ├── scaling_mean.png
│       │   ├── scaling_minmax.png
│       │   ├── scaling_raw.png
│       │   ├── scaling_zscore.png
│       │   ├── sklearn_1d_example.png
│       │   └── vectorization.png
│       ├── neural-network-sketch
│       │   ├── activations.png
│       │   ├── architectures_step1.png
│       │   ├── architectures_step2.png
│       │   ├── architectures_step3.png
│       │   ├── housing_network_step1.png
│       │   ├── housing_network_step2.png
│       │   ├── housing_network_step3.png
│       │   ├── housing_relu.png
│       │   ├── image_to_vector.png
│       │   ├── mnist_digits.png
│       │   ├── neuron_zoom.png
│       │   ├── two_layer_step1.png
│       │   ├── two_layer_step2.png
│       │   ├── two_layer_step3.png
│       │   ├── why_nonlinear.png
│       │   └── zero_vs_one.png
│       ├── overfitting-sketch
│       │   ├── bias_variance.png
│       │   ├── data.png
│       │   ├── fit_classification_step1.png
│       │   ├── fit_classification_step2.png
│       │   ├── fit_classification_step3.png
│       │   ├── fit_regression_step1.png
│       │   ├── fit_regression_step2.png
│       │   ├── fit_regression_step3.png
│       │   ├── lambda_sweep.png
│       │   ├── more_data.png
│       │   ├── regularised_classification.png
│       │   ├── regularised_fit.png
│       │   ├── regularised_weights.png
│       │   ├── validation_degree.png
│       │   └── validation_lambda.png
│       ├── pytorch-sketch
│       │   ├── course_notebooks.png
│       │   ├── graph_step1.png
│       │   ├── graph_step2.png
│       │   ├── hpc_workflow.png
│       │   ├── learning_curve.png
│       │   ├── minibatch_curves.png
│       │   ├── tensor_ranks.png
│       │   └── training_loop.png
│       ├── rnn-sketch
│       │   ├── bidirectional.png
│       │   ├── bptt.png
│       │   ├── deep_rnn.png
│       │   ├── dense_on_sequence.png
│       │   ├── dino_loss.png
│       │   ├── embedding_map.png
│       │   ├── gradient_lstm.png
│       │   ├── gradient_norms.png
│       │   ├── gru.png
│       │   ├── lm_sampling.png
│       │   ├── lm_training.png
│       │   ├── lstm.png
│       │   ├── one_hot.png
│       │   ├── rnn_cell.png
│       │   ├── rnn_types.png
│       │   ├── unrolled_step1.png
│       │   └── unrolled_step2.png
│       └── svm-sketch
│           ├── geometry.png
│           ├── hinge_vs_logistic.png
│           ├── lagrange_equality.png
│           ├── lagrange_inequality.png
│           ├── rbf_grid.png
│           ├── soft_alphas.png
│           ├── svm_kernel.png
│           ├── svm_margin_step1.png
│           ├── svm_margin_step2.png
│           ├── svm_soft_margin.png
│           └── worked_example.png
├── _quarto.yml
├── scripts
│   ├── fonts
│   │   └── xkcd-script.ttf
│   ├── make_figures_lecture01.py
│   ├── make_figures_lecture02.py
│   ├── make_figures_lecture03.py
│   ├── make_figures_lecture04.py
│   ├── make_figures_lecture05.py
│   ├── make_figures_lecture06.py
│   ├── make_figures_lecture07.py
│   ├── make_figures_lecture08.py
│   └── make_figures_lecture09.py
└── styles
    ├── beamer-header.tex
    ├── contact.html
    ├── fonts
    │   ├── atkinson-hyperlegible-latin-400-italic.woff2
    │   ├── atkinson-hyperlegible-latin-400-normal.woff2
    │   ├── atkinson-hyperlegible-latin-700-italic.woff2
    │   └── atkinson-hyperlegible-latin-700-normal.woff2
    ├── fonts.css
    └── lecture.scss

38 directories, 463 files
