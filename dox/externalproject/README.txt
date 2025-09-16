This folder contains an example of an external project that uses code/libraries from elastix. It is meant to be built outside the elastix tree. You need to have compiled elastix first. Then run cmake with this folder as source directory.

Two small test executables will be created:

    elastix_translation_example: a basic registration using the standard NCC metric.

    elastix_impact_metric_example: a variant using the Impact metric.
