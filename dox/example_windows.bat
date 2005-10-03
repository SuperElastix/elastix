
@echo # This batch file shows an example of how to call elastix.
@echo # Run "elastix -help" for more information.
@echo # 
@echo # This example requires elastix to be in your PATH.
@echo 
@echo # Here we perform a registration by doing sequentially:
@echo #  1) a translation-registration
@echo #  2) an affine-registration
@echo #  3) a bspline-registration
@echo # We write all output in the exampleoutput directory.
@echo #

elastix -f exampleinput/fixed.mhd -m exampleinput/moving.mhd  -out exampleoutput -p exampleinput/parameters_Translation.txt -p exampleinput/parameters_Affine.txt -p exampleinput/parameters_BSpline.txt
 
@echo # In the following way we may mask the input images,
@echo # to prevent background pixels to be taken into account:
@echo # 
@echo # elastix -f exampleinput/fixed.mhd -m exampleinput/moving.mhd -fMask exampleinput/mask_fixed.mhd -mMask exampleinput/mask_moving.mhd -out exampleoutput -p exampleinput/parameters_Translation.txt -p exampleinput/parameters_Affine.txt -p exampleinput/parameters_BSpline.txt

@echo #
@echo # See /exampleoutput/ for the results:
@echo #  1) elastix.log = the logfile
@echo #  2) result.0.mhd = the translation transformed result
@echo #  3) result.1.mhd = the affine transformed result
@echo #  4) result.2.mhd = the bspline transformed result
@echo #  5) TransformParameters.0.txt = the found translation
@echo #  6) TransformParameters.1.txt = the found affine parameters
@echo #  7) TransformParameters.2.txt = the found bspline parameters
@echo #
@echo # You may compare result.2.mhd to the solution, which is located in
@echo # ./exampleinput: solution_deformedmovingimage.mhd
@echo #
@echo # After doing the registration, we could apply the found deformation to the moving mask.
@echo # For this we use transformix:
@echo #

transformix -in exampleinput/mask_moving.mhd -out exampleoutput -tp exampleoutput/TransformParameters.2.txt

@echo #
@echo # See /exampleoutput/ for the results:
@echo #  8) transformix.log = the logfile
@echo #  9) result.mhd = the transformed mask
@echo #

@echo exit
