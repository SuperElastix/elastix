/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkTransformRigidityPenaltyTerm_hxx
#define itkTransformRigidityPenaltyTerm_hxx

#include "itkTransformRigidityPenaltyTerm.h"

#include "itkZeroFluxNeumannBoundaryCondition.h"

namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template <class TFixedImage, class TScalarType>
TransformRigidityPenaltyTerm<TFixedImage, TScalarType>::TransformRigidityPenaltyTerm()
{
  /** Weights. */
  this->m_LinearityConditionWeight = NumericTraits<ScalarType>::One;
  this->m_OrthonormalityConditionWeight = NumericTraits<ScalarType>::One;
  this->m_PropernessConditionWeight = NumericTraits<ScalarType>::One;

  /** Values. */
  this->m_RigidityPenaltyTermValue = NumericTraits<MeasureType>::Zero;
  this->m_LinearityConditionValue = NumericTraits<MeasureType>::Zero;
  this->m_OrthonormalityConditionValue = NumericTraits<MeasureType>::Zero;
  this->m_PropernessConditionValue = NumericTraits<MeasureType>::Zero;

  /** Gradient magnitudes. */
  this->m_LinearityConditionGradientMagnitude = NumericTraits<MeasureType>::Zero;
  this->m_OrthonormalityConditionGradientMagnitude = NumericTraits<MeasureType>::Zero;
  this->m_PropernessConditionGradientMagnitude = NumericTraits<MeasureType>::Zero;

  /** Usage. */
  this->m_UseLinearityCondition = true;
  this->m_UseOrthonormalityCondition = true;
  this->m_UsePropernessCondition = true;
  this->m_CalculateLinearityCondition = true;
  this->m_CalculateOrthonormalityCondition = true;
  this->m_CalculatePropernessCondition = true;

  /** Initialize dilation. */
  this->m_DilationRadiusMultiplier = NumericTraits<CoordinateRepresentationType>::One;
  this->m_DilateRigidityImages = true;

  /** Initialize rigidity images and their usage. */
  this->m_UseFixedRigidityImage = true;
  this->m_UseMovingRigidityImage = true;
  this->m_FixedRigidityImage = nullptr;
  this->m_MovingRigidityImage = nullptr;
  this->m_RigidityCoefficientImage = RigidityImageType::New();
  this->m_RigidityCoefficientImageIsFilled = false;

  /** Initialize dilation filter for the rigidity images. */
  this->m_FixedRigidityImageDilation.resize(FixedImageDimension);
  this->m_MovingRigidityImageDilation.resize(MovingImageDimension);
  for (unsigned int i = 0; i < FixedImageDimension; ++i)
  {
    this->m_FixedRigidityImageDilation[i] = nullptr;
    this->m_MovingRigidityImageDilation[i] = nullptr;
  }

  /** Initialize dilated rigidity images. */
  this->m_FixedRigidityImageDilated = nullptr;
  this->m_MovingRigidityImageDilated = nullptr;

  /** We don't use an image sampler for this advanced metric. */
  this->SetUseImageSampler(false);

  this->m_BSplineTransform = nullptr;

} // end Constructor


/**
 * *********************** CheckUseAndCalculationBooleans *****************************
 */

template <class TFixedImage, class TScalarType>
void
TransformRigidityPenaltyTerm<TFixedImage, TScalarType>::CheckUseAndCalculationBooleans()
{
  if (this->m_UseLinearityCondition)
  {
    this->m_CalculateLinearityCondition = true;
  }
  if (this->m_UseOrthonormalityCondition)
  {
    this->m_CalculateOrthonormalityCondition = true;
  }
  if (this->m_UsePropernessCondition)
  {
    this->m_CalculatePropernessCondition = true;
  }

} // end CheckUseAndCalculationBooleans()


/**
 * *********************** Initialize *****************************
 */

template <class TFixedImage, class TScalarType>
void
TransformRigidityPenaltyTerm<TFixedImage, TScalarType>::Initialize()
{
  /** Call the initialize of the superclass. */
  this->Superclass::Initialize();

  /** Check if this transform is a B-spline transform. */
  typename BSplineTransformType::Pointer localBSplineTransform; // default-constructed (null)
  bool                                   transformIsBSpline = this->CheckForBSplineTransform2(localBSplineTransform);
  if (transformIsBSpline)
  {
    this->SetBSplineTransform(localBSplineTransform);
  }

  /** Set the B-spline transform to m_RigidityPenaltyTermMetric. */
  if (!transformIsBSpline)
  {
    itkExceptionMacro(<< "ERROR: this metric expects a B-spline transform.");
  }

  /** Allocate the RigidityCoefficientImage, so that it matches the B-spline grid.
   * Only because the Initialize()-function above is called before,
   * this code is valid, because there the B-spline transform is set.
   */
  RigidityImageRegionType region;
  region.SetSize(localBSplineTransform->GetGridRegion().GetSize());
  region.SetIndex(localBSplineTransform->GetGridRegion().GetIndex());
  this->m_RigidityCoefficientImage->SetRegions(region);
  this->m_RigidityCoefficientImage->SetSpacing(localBSplineTransform->GetGridSpacing());
  this->m_RigidityCoefficientImage->SetOrigin(localBSplineTransform->GetGridOrigin());
  this->m_RigidityCoefficientImage->SetDirection(localBSplineTransform->GetGridDirection());
  this->m_RigidityCoefficientImage->Allocate();

  if (!this->m_UseFixedRigidityImage && !this->m_UseMovingRigidityImage)
  {
    /** Fill the rigidity coefficient image with ones. */
    this->m_RigidityCoefficientImage->FillBuffer(1.0);
  }
  else
  {
    this->DilateRigidityImages();
  }

  /** Reset the filling bool. */
  this->m_RigidityCoefficientImageIsFilled = false;

} // end Initialize()


/**
 * **************** DilateRigidityImages *****************
 */

template <class TFixedImage, class TScalarType>
void
TransformRigidityPenaltyTerm<TFixedImage, TScalarType>::DilateRigidityImages()
{
  /** Dilate m_FixedRigidityImage and m_MovingRigidityImage. */
  if (this->m_DilateRigidityImages)
  {
    /** Some declarations. */
    SERadiusType                        radius;
    std::vector<StructuringElementType> structuringElement(FixedImageDimension);

    /** Setup the pipeline. */
    if (this->m_UseFixedRigidityImage)
    {
      /** Create the dilation filters for the fixedRigidityImage. */
      for (unsigned int i = 0; i < FixedImageDimension; ++i)
      {
        this->m_FixedRigidityImageDilation[i] = DilateFilterType::New();
      }
      this->m_FixedRigidityImageDilation[0]->SetInput(this->m_FixedRigidityImage);
    }
    if (this->m_UseMovingRigidityImage)
    {
      /** Create the dilation filter for the movingRigidityImage. */
      for (unsigned int i = 0; i < FixedImageDimension; ++i)
      {
        this->m_MovingRigidityImageDilation[i] = DilateFilterType::New();
      }
      this->m_MovingRigidityImageDilation[0]->SetInput(this->m_MovingRigidityImage);
    }

    /** Get the B-spline grid spacing. */
    GridSpacingType gridSpacing;
    if (this->m_BSplineTransform.IsNotNull())
    {
      gridSpacing = this->m_BSplineTransform->GetGridSpacing();
    }

    /** Set stuff for the separate dilation. */
    for (unsigned int i = 0; i < FixedImageDimension; ++i)
    {
      /** Create the structuring element. */
      radius.Fill(0);
      radius.SetElement(i, static_cast<unsigned long>(this->m_DilationRadiusMultiplier * gridSpacing[i]));

      structuringElement[i].SetRadius(radius);
      structuringElement[i].CreateStructuringElement();

      /** Set the kernel into all dilation filters.
       * The SetKernel() is implemented using a itkSetMacro, so a
       * this->Modified() is automatically called, which is important,
       * since this changes every time Initialize() is called (every resolution).
       */
      if (this->m_UseFixedRigidityImage)
      {
        this->m_FixedRigidityImageDilation[i]->SetKernel(structuringElement[i]);
      }
      if (this->m_UseMovingRigidityImage)
      {
        this->m_MovingRigidityImageDilation[i]->SetKernel(structuringElement[i]);
      }

      /** Connect the pipelines. */
      if (i > 0)
      {
        if (this->m_UseFixedRigidityImage)
        {
          this->m_FixedRigidityImageDilation[i]->SetInput(this->m_FixedRigidityImageDilation[i - 1]->GetOutput());
        }
        if (this->m_UseMovingRigidityImage)
        {
          this->m_MovingRigidityImageDilation[i]->SetInput(this->m_MovingRigidityImageDilation[i - 1]->GetOutput());
        }
      }
    } // end for loop

    /** Do the dilation for m_FixedRigidityImage. */
    if (this->m_UseFixedRigidityImage)
    {
      try
      {
        this->m_FixedRigidityImageDilation[FixedImageDimension - 1]->Update();
      }
      catch (itk::ExceptionObject & excp)
      {
        /** Add information to the exception. */
        excp.SetLocation("TransformRigidityPenaltyTerm - Initialize()");
        std::string err_str = excp.GetDescription();
        err_str += "\nError while dilating m_FixedRigidityImage.\n";
        excp.SetDescription(err_str);
        /** Pass the exception to an higher level. */
        throw;
      }
    }

    /** Do the dilation for m_MovingRigidityImage. */
    if (this->m_UseMovingRigidityImage)
    {
      try
      {
        this->m_MovingRigidityImageDilation[MovingImageDimension - 1]->Update();
      }
      catch (itk::ExceptionObject & excp)
      {
        /** Add information to the exception. */
        excp.SetLocation("TransformRigidityPenaltyTerm - Initialize()");
        std::string err_str = excp.GetDescription();
        err_str += "\nError while dilating m_MovingRigidityImage.\n";
        excp.SetDescription(err_str);
        /** Pass the exception to an higher level. */
        throw;
      }
    }

    /** Put the output of the dilation into some dilated images. */
    if (this->m_UseFixedRigidityImage)
    {
      this->m_FixedRigidityImageDilated = this->m_FixedRigidityImageDilation[FixedImageDimension - 1]->GetOutput();
    }
    if (this->m_UseMovingRigidityImage)
    {
      this->m_MovingRigidityImageDilated = this->m_MovingRigidityImageDilation[MovingImageDimension - 1]->GetOutput();
    }
  } // end if rigidity images should be dilated
  else
  {
    /** Copy the pointers of the undilated images to the dilated ones
     * if no dilation is needed.
     */
    if (this->m_UseFixedRigidityImage)
    {
      this->m_FixedRigidityImageDilated = this->m_FixedRigidityImage;
    }
    if (this->m_UseMovingRigidityImage)
    {
      this->m_MovingRigidityImageDilated = this->m_MovingRigidityImage;
    }

  } // end else if

} // end DilateRigidityImages()


/**
 * **************** FillRigidityCoefficientImage *****************
 */

template <class TFixedImage, class TScalarType>
void
TransformRigidityPenaltyTerm<TFixedImage, TScalarType>::FillRigidityCoefficientImage(
  const ParametersType & parameters) const
{
  /** Sanity check. */
  if (!this->m_UseFixedRigidityImage && !this->m_UseMovingRigidityImage)
  {
    return;
  }

  /** The rigidity image only changes when it depends on the moving image. */
  if (!this->m_UseMovingRigidityImage && this->m_RigidityCoefficientImageIsFilled)
  {
    return;
  }

  /** Make sure that the transform is up to date. */
  this->m_Transform->SetParameters(parameters);

  /** Create and reset an iterator over m_RigidityCoefficientImage. */
  RigidityImageIteratorType it(this->m_RigidityCoefficientImage,
                               this->m_RigidityCoefficientImage->GetLargestPossibleRegion());
  it.GoToBegin();

  /** Fill m_RigidityCoefficientImage. */
  RigidityPixelType      fixedValue, movingValue, in;
  RigidityImagePointType point{};
  RigidityImageIndexType index1, index2;
  index1.Fill(0);
  index2.Fill(0);
  fixedValue = NumericTraits<RigidityPixelType>::Zero;
  movingValue = NumericTraits<RigidityPixelType>::Zero;
  in = NumericTraits<RigidityPixelType>::Zero;
  bool isInFixedImage = false;
  bool isInMovingImage = false;
  while (!it.IsAtEnd())
  {
    /** Get current pixel in world coordinates. */
    this->m_RigidityCoefficientImage->TransformIndexToPhysicalPoint(it.GetIndex(), point);

    /** Get the corresponding indices in the fixed and moving RigidityImage's.
     * NOTE: Floating point index results are truncated to integers.
     */
    if (this->m_UseFixedRigidityImage)
    {
      isInFixedImage = this->m_FixedRigidityImageDilated->TransformPhysicalPointToIndex(point, index1);
      // \todo: Note that we should actually use the inverted initial transform
      // here, a little bit like:
      // isInFixedImage = this->m_FixedRigidityImageDilated
      //   ->TransformPhysicalPointToIndex( this->Transform->GetInitialTransform()
      //   ->GetInverse()->TransformPoint( point ), index1 );
      // This is needed to compensate for the B-spline grid shift that has been
      // performed earlier, which causes the B-spline grid region and thus the
      // m_RigidityCoefficientImage region to be different from the fixed (coefffient)
      // image region.
      //
      // Since in general the inverse does not exist, alternative strategies may be:
      // 1) Approximate the inverse of the initial transform using inverse deformation
      //    field approximation filters available in the ITK
      // 2) Instead op looping over m_RigidityCoefficientImage, we can loop over
      //    m_FixedRigidityImageDilated, employ the normal forward initial transform,
      //    and fill m_RigidityCoefficientImage this way. A downside is that holes may
      //    be created in the m_RigidityCoefficientImage, although this has low
      //    likelihood, since the resolution of m_RigidityCoefficientImage is much
      //    lower than the fixed (rigidity) image. And we could check for these holes
      //    afterwards.
      // WARNING: So, currently the rigidity penalty term does not correctly support
      // initial transforms, in case a fixed coefficient image is provided. It works
      // correctly if only a moving coefficient image is provided.
      // Perhaps we should remove the option to supply the fixed coefficient image,
      // since the moving one should really be used.
    }
    if (this->m_UseMovingRigidityImage)
    {
      isInMovingImage = this->m_MovingRigidityImageDilated->TransformPhysicalPointToIndex(
        // this->m_Transform->TransformPoint( point ), index2 );
        this->m_BSplineTransform->TransformPoint(point),
        index2);
    }

    /** Get the values at those positions. */
    if (this->m_UseFixedRigidityImage)
    {
      if (isInFixedImage)
      {
        fixedValue = this->m_FixedRigidityImageDilated->GetPixel(index1);
      }
      else
      {
        fixedValue = 0.0;
      }
    }

    if (this->m_UseMovingRigidityImage)
    {
      if (isInMovingImage)
      {
        movingValue = this->m_MovingRigidityImageDilated->GetPixel(index2);
      }
      else
      {
        movingValue = 0.0;
      }
    }

    /** Determine the maximum. */
    if (this->m_UseFixedRigidityImage && this->m_UseMovingRigidityImage)
    {
      in = (fixedValue > movingValue ? fixedValue : movingValue);
    }
    else if (this->m_UseFixedRigidityImage && !this->m_UseMovingRigidityImage)
    {
      in = fixedValue;
    }
    else if (!this->m_UseFixedRigidityImage && this->m_UseMovingRigidityImage)
    {
      in = movingValue;
    }
    /** else{} is not happening here, because we assume that one of them is true.
     * In our case we checked that in the derived class: elxMattesMIWRR.
     */

    /** Set it. */
    it.Set(in);

    /** Increase iterator. */
    ++it;
  } // end while loop over rigidity coefficient image

  /** Remember that the rigidity coefficient image is filled. */
  this->m_RigidityCoefficientImageIsFilled = true;

} // end FillRigidityCoefficientImage()


/**
 * *********************** GetValue *****************************
 */

template <class TFixedImage, class TScalarType>
auto
TransformRigidityPenaltyTerm<TFixedImage, TScalarType>::GetValue(const ParametersType & parameters) const -> MeasureType
{
  /** Fill the rigidity image based on the current transform parameters. */
  this->FillRigidityCoefficientImage(parameters);

  /** Set output values to zero. */
  this->m_RigidityPenaltyTermValue = NumericTraits<MeasureType>::Zero;
  this->m_LinearityConditionValue = NumericTraits<MeasureType>::Zero;
  this->m_OrthonormalityConditionValue = NumericTraits<MeasureType>::Zero;
  this->m_PropernessConditionValue = NumericTraits<MeasureType>::Zero;

  /** Set the parameters in the transform.
   * In this function, also the coefficient images are created.
   */
  this->m_BSplineTransform->SetParameters(parameters);

  /** Sanity check. */
  if (ImageDimension != 2 && ImageDimension != 3)
  {
    itkExceptionMacro(<< "ERROR: This filter is only implemented for dimension 2 and 3.");
  }

  /** Get a handle to the B-spline coefficient images. */
  std::vector<CoefficientImagePointer> inputImages(ImageDimension);
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    inputImages[i] = this->m_BSplineTransform->GetCoefficientImages()[i];
  }

  /** Get the B-spline coefficient image spacing. */
  CoefficientImageSpacingType spacing = inputImages[0]->GetSpacing();

  /** TASK 0:
   * Compute the rigidityCoefficientSum and check on it.
   *
   ************************************************************************* */

  /** Create iterator over the rigidity coeficient image. */
  CoefficientImageIteratorType it_RCI(this->m_RigidityCoefficientImage,
                                      this->m_RigidityCoefficientImage->GetLargestPossibleRegion());
  it_RCI.GoToBegin();
  ScalarType rigidityCoefficientSum = NumericTraits<ScalarType>::Zero;

  /** Add the rigidity coefficients together. */
  while (!it_RCI.IsAtEnd())
  {
    rigidityCoefficientSum += it_RCI.Get();
    ++it_RCI;
  }

  /** Check for early termination. */
  if (rigidityCoefficientSum < 1e-14)
  {
    this->m_RigidityPenaltyTermValue = NumericTraits<MeasureType>::Zero;
    return this->m_RigidityPenaltyTermValue;
  }

  /** TASK 1:
   * Prepare for the calculation of the rigidity penalty term.
   *
   ************************************************************************* */

  /** Create 1D neighbourhood operators. */
  std::vector<NeighborhoodType> Operators_A(ImageDimension), Operators_B(ImageDimension), Operators_C(ImageDimension),
    Operators_D(ImageDimension), Operators_E(ImageDimension), Operators_F(ImageDimension), Operators_G(ImageDimension),
    Operators_H(ImageDimension), Operators_I(ImageDimension);

  /** Create B-spline coefficient images that are filtered once. */
  std::vector<CoefficientImagePointer> ui_FA(ImageDimension), ui_FB(ImageDimension), ui_FC(ImageDimension),
    ui_FD(ImageDimension), ui_FE(ImageDimension), ui_FF(ImageDimension), ui_FG(ImageDimension), ui_FH(ImageDimension),
    ui_FI(ImageDimension);

  /** For all dimensions ... */
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    /** ... create the filtered images ... */
    ui_FA[i] = CoefficientImageType::New();
    ui_FB[i] = CoefficientImageType::New();
    ui_FD[i] = CoefficientImageType::New();
    ui_FE[i] = CoefficientImageType::New();
    ui_FG[i] = CoefficientImageType::New();
    if (ImageDimension == 3)
    {
      ui_FC[i] = CoefficientImageType::New();
      ui_FF[i] = CoefficientImageType::New();
      ui_FH[i] = CoefficientImageType::New();
      ui_FI[i] = CoefficientImageType::New();
    }
    /** ... and the apropiate operators.
     * The operators C, D and E from the paper are here created
     * by Create1DOperator D, E and G, because of the 3D case and history.
     */
    this->Create1DOperator(Operators_A[i], "FA_xi", i + 1, spacing);
    this->Create1DOperator(Operators_B[i], "FB_xi", i + 1, spacing);
    this->Create1DOperator(Operators_D[i], "FD_xi", i + 1, spacing);
    this->Create1DOperator(Operators_E[i], "FE_xi", i + 1, spacing);
    this->Create1DOperator(Operators_G[i], "FG_xi", i + 1, spacing);
    if (ImageDimension == 3)
    {
      this->Create1DOperator(Operators_C[i], "FC_xi", i + 1, spacing);
      this->Create1DOperator(Operators_F[i], "FF_xi", i + 1, spacing);
      this->Create1DOperator(Operators_H[i], "FH_xi", i + 1, spacing);
      this->Create1DOperator(Operators_I[i], "FI_xi", i + 1, spacing);
    }
  } // end for loop

  /** TASK 2:
   * Filter the B-spline coefficient images.
   *
   ************************************************************************* */

  /** Filter the inputImages. */
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    ui_FA[i] = this->FilterSeparable(inputImages[i], Operators_A);
    ui_FB[i] = this->FilterSeparable(inputImages[i], Operators_B);
    ui_FD[i] = this->FilterSeparable(inputImages[i], Operators_D);
    ui_FE[i] = this->FilterSeparable(inputImages[i], Operators_E);
    ui_FG[i] = this->FilterSeparable(inputImages[i], Operators_G);
    if (ImageDimension == 3)
    {
      ui_FC[i] = this->FilterSeparable(inputImages[i], Operators_C);
      ui_FF[i] = this->FilterSeparable(inputImages[i], Operators_F);
      ui_FH[i] = this->FilterSeparable(inputImages[i], Operators_H);
      ui_FI[i] = this->FilterSeparable(inputImages[i], Operators_I);
    }
  }

  /** TASK 3:
   * Create iterators.
   *
   ************************************************************************* */

  /** Create iterators over ui_F?. */
  std::vector<CoefficientImageIteratorType> itA(ImageDimension), itB(ImageDimension), itC(ImageDimension),
    itD(ImageDimension), itE(ImageDimension), itF(ImageDimension), itG(ImageDimension), itH(ImageDimension),
    itI(ImageDimension);

  /** Create iterators. */
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    /** Create iterators. */
    itA[i] = CoefficientImageIteratorType(ui_FA[i], ui_FA[i]->GetLargestPossibleRegion());
    itB[i] = CoefficientImageIteratorType(ui_FB[i], ui_FB[i]->GetLargestPossibleRegion());
    itD[i] = CoefficientImageIteratorType(ui_FD[i], ui_FD[i]->GetLargestPossibleRegion());
    itE[i] = CoefficientImageIteratorType(ui_FE[i], ui_FE[i]->GetLargestPossibleRegion());
    itG[i] = CoefficientImageIteratorType(ui_FG[i], ui_FG[i]->GetLargestPossibleRegion());
    if (ImageDimension == 3)
    {
      itC[i] = CoefficientImageIteratorType(ui_FC[i], ui_FC[i]->GetLargestPossibleRegion());
      itF[i] = CoefficientImageIteratorType(ui_FF[i], ui_FF[i]->GetLargestPossibleRegion());
      itH[i] = CoefficientImageIteratorType(ui_FH[i], ui_FH[i]->GetLargestPossibleRegion());
      itI[i] = CoefficientImageIteratorType(ui_FI[i], ui_FI[i]->GetLargestPossibleRegion());
    }
    /** Reset iterators. */
    itA[i].GoToBegin();
    itB[i].GoToBegin();
    itD[i].GoToBegin();
    itE[i].GoToBegin();
    itG[i].GoToBegin();
    if (ImageDimension == 3)
    {
      itC[i].GoToBegin();
      itF[i].GoToBegin();
      itH[i].GoToBegin();
      itI[i].GoToBegin();
    }
  }

  /** TASK 4A:
   * Do the actual calculation of the rigidity penalty term value.
   * Calculate the orthonormality term.
   *
   ************************************************************************* */

  /** Reset all iterators. */
  it_RCI.GoToBegin();

  if (this->m_CalculateOrthonormalityCondition)
  {
    ScalarType mu1_A, mu2_A, mu3_A, mu1_B, mu2_B, mu3_B, mu1_C, mu2_C, mu3_C;
    while (!itA[0].IsAtEnd())
    {
      /** Copy values: this way we avoid calling Get() so many times.
       * It also improves code readability.
       */
      mu1_A = itA[0].Get();
      mu2_A = itA[1].Get();
      mu1_B = itB[0].Get();
      mu2_B = itB[1].Get();
      if (ImageDimension == 3)
      {
        mu3_A = itA[2].Get();
        mu3_B = itB[2].Get();
        mu1_C = itC[0].Get();
        mu2_C = itC[1].Get();
        mu3_C = itC[2].Get();
      }

      if (ImageDimension == 2)
      {
        this->m_OrthonormalityConditionValue +=
          it_RCI.Get() * (std::pow(+(1.0 + mu1_A) * (1.0 + mu1_A) + mu2_A * mu2_A - 1.0, 2.0) +
                          std::pow(+mu1_B * mu1_B + (1.0 + mu2_B) * (1.0 + mu2_B) - 1.0, 2.0) +
                          std::pow(+(1.0 + mu1_A) * mu1_B + mu2_A * (1.0 + mu2_B), 2.0));
      }
      else if (ImageDimension == 3)
      {
        this->m_OrthonormalityConditionValue +=
          it_RCI.Get() * (std::pow(+(1.0 + mu1_A) * (1.0 + mu1_A) + mu2_A * mu2_A + mu3_A * mu3_A - 1.0, 2.0) +
                          std::pow(+(1.0 + mu1_A) * mu1_B + mu2_A * (1.0 + mu2_B) + mu3_A * mu3_B, 2.0) +
                          std::pow(+(1.0 + mu1_A) * mu1_C + mu2_A * mu2_C + mu3_A * (1.0 + mu3_C), 2.0) +
                          std::pow(+mu1_B * mu1_B + (1.0 + mu2_B) * (1.0 + mu2_B) + mu3_B * mu3_B - 1.0, 2.0) +
                          std::pow(+mu1_B * mu1_C + (1.0 + mu2_B) * mu2_C + mu3_B * (1.0 + mu3_C), 2.0) +
                          std::pow(+mu1_C * mu1_C + mu2_C * mu2_C + (1.0 + mu3_C) * (1.0 + mu3_C) - 1.0, 2.0));
      }

      /** Increase all iterators. */
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        ++itA[i];
        ++itB[i];
        if (ImageDimension == 3)
        {
          ++itC[i];
        }
      }
      ++it_RCI;
    } // end while
  }   // end if do orthonormality

  /** TASK 4B:
   * Do the actual calculation of the rigidity penalty term value.
   * Calculate the properness term.
   *
   ************************************************************************* */

  /** Reset all iterators. */
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    itA[i].GoToBegin();
    itB[i].GoToBegin();
    if (ImageDimension == 3)
    {
      itC[i].GoToBegin();
    }
  }
  it_RCI.GoToBegin();

  if (this->m_CalculatePropernessCondition)
  {
    ScalarType mu1_A, mu2_A, mu3_A, mu1_B, mu2_B, mu3_B, mu1_C, mu2_C, mu3_C;
    while (!itA[0].IsAtEnd())
    {
      /** Copy values: this way we avoid calling Get() so many times.
       * It also improves code readability.
       */
      mu1_A = itA[0].Get();
      mu2_A = itA[1].Get();
      mu1_B = itB[0].Get();
      mu2_B = itB[1].Get();
      if (ImageDimension == 3)
      {
        mu3_A = itA[2].Get();
        mu3_B = itB[2].Get();
        mu1_C = itC[0].Get();
        mu2_C = itC[1].Get();
        mu3_C = itC[2].Get();
      }

      if (ImageDimension == 2)
      {
        this->m_PropernessConditionValue +=
          it_RCI.Get() * (std::pow(+(1.0 + mu1_A) * (1.0 + mu2_B) - mu2_A * mu1_B - 1.0, 2.0));
      }
      else if (ImageDimension == 3)
      {
        this->m_PropernessConditionValue +=
          it_RCI.Get() * (std::pow(-mu1_C * (1.0 + mu2_B) * mu3_A + mu1_B * mu2_C * mu3_A + mu1_C * mu2_A * mu3_B -
                                     (1.0 + mu1_A) * mu2_C * mu3_B - mu1_B * mu2_A * (1.0 + mu3_C) +
                                     (1.0 + mu1_A) * (1.0 + mu2_B) * (1.0 + mu3_C) - 1.0,
                                   2.0));
      }

      /** Increase all iterators. */
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        ++itA[i];
        ++itB[i];
        if (ImageDimension == 3)
        {
          ++itC[i];
        }
      }
      ++it_RCI;

    } // end while
  }   // end if do properness

  /** TASK 4C:
   * Do the actual calculation of the rigidity penalty term value.
   * Calculate the linearity term.
   *
   ************************************************************************* */

  /** Reset all iterators. */
  it_RCI.GoToBegin();

  if (this->m_CalculateLinearityCondition)
  {
    while (!itD[0].IsAtEnd())
    {
      /** Linearity condition part. */
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        this->m_LinearityConditionValue +=
          it_RCI.Get() * (+itD[i].Get() * itD[i].Get() + itE[i].Get() * itE[i].Get() + itG[i].Get() * itG[i].Get());
        if (ImageDimension == 3)
        {
          this->m_LinearityConditionValue +=
            it_RCI.Get() * (+itF[i].Get() * itF[i].Get() + itH[i].Get() * itH[i].Get() + itI[i].Get() * itI[i].Get());
        }
      } // end loop over i

      /** Increase all iterators. */
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        ++itD[i];
        ++itE[i];
        ++itG[i];
        if (ImageDimension == 3)
        {
          ++itF[i];
          ++itH[i];
          ++itI[i];
        }
      }
      ++it_RCI;

    } // end while
  }   // end if do properness

  /** TASK 5:
   * Do the actual calculation of the rigidity penalty term value.
   *
   ************************************************************************* */

  /** Calculate the rigidity penalty term value. */
  if (this->m_CalculateLinearityCondition)
  {
    this->m_LinearityConditionValue /= rigidityCoefficientSum;
  }
  if (this->m_CalculateOrthonormalityCondition)
  {
    this->m_OrthonormalityConditionValue /= rigidityCoefficientSum;
  }
  if (this->m_CalculatePropernessCondition)
  {
    this->m_PropernessConditionValue /= rigidityCoefficientSum;
  }

  if (this->m_UseLinearityCondition)
  {
    this->m_RigidityPenaltyTermValue += this->m_LinearityConditionWeight * this->m_LinearityConditionValue;
  }
  if (this->m_UseOrthonormalityCondition)
  {
    this->m_RigidityPenaltyTermValue += this->m_OrthonormalityConditionWeight * this->m_OrthonormalityConditionValue;
  }
  if (this->m_UsePropernessCondition)
  {
    this->m_RigidityPenaltyTermValue += this->m_PropernessConditionWeight * this->m_PropernessConditionValue;
  }

  /** Return the rigidity penalty term value. */
  return this->m_RigidityPenaltyTermValue;

} // end GetValue()


/**
 * *********************** GetDerivative ************************
 */

template <class TFixedImage, class TScalarType>
void
TransformRigidityPenaltyTerm<TFixedImage, TScalarType>::GetDerivative(const ParametersType & parameters,
                                                                      DerivativeType &       derivative) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue = NumericTraits<MeasureType>::Zero;
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()


/**
 * *********************** BeforeThreadedGetValueAndDerivative ***********************
 */

template <class TFixedImage, class TScalarType>
void
TransformRigidityPenaltyTerm<TFixedImage, TScalarType>::BeforeThreadedGetValueAndDerivative(
  const TransformParametersType & parameters) const
{
  /** In this function do all stuff that cannot be multi-threaded.
   * Meant for use in the combo-metric. So, I did not think about general usage yet.
   */
  if (this->m_UseMetricSingleThreaded)
  {
    this->m_BSplineTransform->SetParameters(parameters);
  }

} // end BeforeThreadedGetValueAndDerivative()


/**
 * *********************** GetValueAndDerivative ****************
 */

template <class TFixedImage, class TScalarType>
void
TransformRigidityPenaltyTerm<TFixedImage, TScalarType>::GetValueAndDerivative(const ParametersType & parameters,
                                                                              MeasureType &          value,
                                                                              DerivativeType &       derivative) const
{
  /** Fill the rigidity image based on the current transform parameters. */
  this->FillRigidityCoefficientImage(parameters);

  /** Set output values to zero. */
  value = NumericTraits<MeasureType>::Zero;
  this->m_RigidityPenaltyTermValue = NumericTraits<MeasureType>::Zero;
  this->m_LinearityConditionValue = NumericTraits<MeasureType>::Zero;
  this->m_OrthonormalityConditionValue = NumericTraits<MeasureType>::Zero;
  this->m_PropernessConditionValue = NumericTraits<MeasureType>::Zero;

  /** Set output values to zero. */
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(NumericTraits<MeasureType>::ZeroValue());

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Sanity check. */
  if (ImageDimension != 2 && ImageDimension != 3)
  {
    itkExceptionMacro(<< "ERROR: This filter is only implemented for dimension 2 and 3.");
  }

  /** Get a handle to the B-spline coefficient images. */
  std::vector<CoefficientImagePointer> inputImages(ImageDimension);
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    inputImages[i] = this->m_BSplineTransform->GetCoefficientImages()[i];
  }

  /** Get the B-spline coefficient image spacing. */
  CoefficientImageSpacingType spacing = inputImages[0]->GetSpacing();

  /** TASK 0:
   * Compute the rigidityCoefficientSum and check on it.
   *
   ************************************************************************* */

  /** Create iterator over the rigidity coeficient image. */
  CoefficientImageIteratorType it_RCI(this->m_RigidityCoefficientImage,
                                      this->m_RigidityCoefficientImage->GetLargestPossibleRegion());
  it_RCI.GoToBegin();
  ScalarType rigidityCoefficientSum = NumericTraits<ScalarType>::Zero;

  /** Add the rigidity coefficients together. */
  while (!it_RCI.IsAtEnd())
  {
    rigidityCoefficientSum += it_RCI.Get();
    ++it_RCI;
  }

  /** Check for early termination. */
  if (rigidityCoefficientSum < 1e-14)
  {
    this->m_RigidityPenaltyTermValue = NumericTraits<MeasureType>::Zero;
    return;
  }

  /** TASK 1:
   * Prepare for the calculation of the rigidity penalty term.
   *
   ************************************************************************* */

  /** Create 1D neighbourhood operators. */
  std::vector<NeighborhoodType> Operators_A(ImageDimension), Operators_B(ImageDimension), Operators_C(ImageDimension),
    Operators_D(ImageDimension), Operators_E(ImageDimension), Operators_F(ImageDimension), Operators_G(ImageDimension),
    Operators_H(ImageDimension), Operators_I(ImageDimension);

  /** Create B-spline coefficient images that are filtered once. */
  std::vector<CoefficientImagePointer> ui_FA(ImageDimension), ui_FB(ImageDimension), ui_FC(ImageDimension),
    ui_FD(ImageDimension), ui_FE(ImageDimension), ui_FF(ImageDimension), ui_FG(ImageDimension), ui_FH(ImageDimension),
    ui_FI(ImageDimension);

  /** For all dimensions ... */
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    /** ... create the filtered images ... */
    ui_FA[i] = CoefficientImageType::New();
    ui_FB[i] = CoefficientImageType::New();
    ui_FD[i] = CoefficientImageType::New();
    ui_FE[i] = CoefficientImageType::New();
    ui_FG[i] = CoefficientImageType::New();
    if (ImageDimension == 3)
    {
      ui_FC[i] = CoefficientImageType::New();
      ui_FF[i] = CoefficientImageType::New();
      ui_FH[i] = CoefficientImageType::New();
      ui_FI[i] = CoefficientImageType::New();
    }
    /** ... and the apropiate operators.
     * The operators C, D and E from the paper are here created
     * by Create1DOperator D, E and G, because of the 3D case and history.
     */
    this->Create1DOperator(Operators_A[i], "FA_xi", i + 1, spacing);
    this->Create1DOperator(Operators_B[i], "FB_xi", i + 1, spacing);
    this->Create1DOperator(Operators_D[i], "FD_xi", i + 1, spacing);
    this->Create1DOperator(Operators_E[i], "FE_xi", i + 1, spacing);
    this->Create1DOperator(Operators_G[i], "FG_xi", i + 1, spacing);
    if (ImageDimension == 3)
    {
      this->Create1DOperator(Operators_C[i], "FC_xi", i + 1, spacing);
      this->Create1DOperator(Operators_F[i], "FF_xi", i + 1, spacing);
      this->Create1DOperator(Operators_H[i], "FH_xi", i + 1, spacing);
      this->Create1DOperator(Operators_I[i], "FI_xi", i + 1, spacing);
    }
  } // end for loop

  /** TASK 2:
   * Filter the B-spline coefficient images.
   *
   ************************************************************************* */

  /** Filter the inputImages. */
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    ui_FA[i] = this->FilterSeparable(inputImages[i], Operators_A);
    ui_FB[i] = this->FilterSeparable(inputImages[i], Operators_B);
    ui_FD[i] = this->FilterSeparable(inputImages[i], Operators_D);
    ui_FE[i] = this->FilterSeparable(inputImages[i], Operators_E);
    ui_FG[i] = this->FilterSeparable(inputImages[i], Operators_G);
    if (ImageDimension == 3)
    {
      ui_FC[i] = this->FilterSeparable(inputImages[i], Operators_C);
      ui_FF[i] = this->FilterSeparable(inputImages[i], Operators_F);
      ui_FH[i] = this->FilterSeparable(inputImages[i], Operators_H);
      ui_FI[i] = this->FilterSeparable(inputImages[i], Operators_I);
    }
  }

  /** TASK 3:
   * Create subparts and iterators.
   *
   ************************************************************************* */

  /** Create iterators over ui_F?. */
  std::vector<CoefficientImageIteratorType> itA(ImageDimension), itB(ImageDimension), itC(ImageDimension),
    itD(ImageDimension), itE(ImageDimension), itF(ImageDimension), itG(ImageDimension), itH(ImageDimension),
    itI(ImageDimension);

  /** Create iterators. */
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    /** Create iterators. */
    itA[i] = CoefficientImageIteratorType(ui_FA[i], ui_FA[i]->GetLargestPossibleRegion());
    itB[i] = CoefficientImageIteratorType(ui_FB[i], ui_FB[i]->GetLargestPossibleRegion());
    itD[i] = CoefficientImageIteratorType(ui_FD[i], ui_FD[i]->GetLargestPossibleRegion());
    itE[i] = CoefficientImageIteratorType(ui_FE[i], ui_FE[i]->GetLargestPossibleRegion());
    itG[i] = CoefficientImageIteratorType(ui_FG[i], ui_FG[i]->GetLargestPossibleRegion());
    if (ImageDimension == 3)
    {
      itC[i] = CoefficientImageIteratorType(ui_FC[i], ui_FC[i]->GetLargestPossibleRegion());
      itF[i] = CoefficientImageIteratorType(ui_FF[i], ui_FF[i]->GetLargestPossibleRegion());
      itH[i] = CoefficientImageIteratorType(ui_FH[i], ui_FH[i]->GetLargestPossibleRegion());
      itI[i] = CoefficientImageIteratorType(ui_FI[i], ui_FI[i]->GetLargestPossibleRegion());
    }
    /** Reset iterators. */
    itA[i].GoToBegin();
    itB[i].GoToBegin();
    itD[i].GoToBegin();
    itE[i].GoToBegin();
    itG[i].GoToBegin();
    if (ImageDimension == 3)
    {
      itC[i].GoToBegin();
      itF[i].GoToBegin();
      itH[i].GoToBegin();
      itI[i].GoToBegin();
    }
  }

  /** Create orthonormality and properness parts. */
  std::vector<std::vector<CoefficientImagePointer>> OCparts(ImageDimension);
  std::vector<std::vector<CoefficientImagePointer>> PCparts(ImageDimension);
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    OCparts[i].resize(ImageDimension);
    PCparts[i].resize(ImageDimension);
    for (unsigned int j = 0; j < ImageDimension; ++j)
    {
      OCparts[i][j] = CoefficientImageType::New();
      OCparts[i][j]->SetRegions(inputImages[0]->GetLargestPossibleRegion());
      OCparts[i][j]->Allocate();
      PCparts[i][j] = CoefficientImageType::New();
      PCparts[i][j]->SetRegions(inputImages[0]->GetLargestPossibleRegion());
      PCparts[i][j]->Allocate();
    }
  }

  /** Create linearity parts. */
  unsigned int                                      NofLParts = 3 * ImageDimension - 3;
  std::vector<std::vector<CoefficientImagePointer>> LCparts(ImageDimension);
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    LCparts[i].resize(NofLParts);
    for (unsigned int j = 0; j < NofLParts; ++j)
    {
      LCparts[i][j] = CoefficientImageType::New();
      LCparts[i][j]->SetRegions(inputImages[0]->GetLargestPossibleRegion());
      LCparts[i][j]->Allocate();
    }
  }

  /** Create iterators over all parts. */
  std::vector<std::vector<CoefficientImageIteratorType>> itOCp(ImageDimension);
  std::vector<std::vector<CoefficientImageIteratorType>> itPCp(ImageDimension);
  std::vector<std::vector<CoefficientImageIteratorType>> itLCp(ImageDimension);
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    itOCp[i].resize(ImageDimension);
    itPCp[i].resize(ImageDimension);
    itLCp[i].resize(NofLParts);
    for (unsigned int j = 0; j < ImageDimension; ++j)
    {
      itOCp[i][j] = CoefficientImageIteratorType(OCparts[i][j], OCparts[i][j]->GetLargestPossibleRegion());
      itOCp[i][j].GoToBegin();
      itPCp[i][j] = CoefficientImageIteratorType(PCparts[i][j], PCparts[i][j]->GetLargestPossibleRegion());
      itPCp[i][j].GoToBegin();
    }
    for (unsigned int j = 0; j < NofLParts; ++j)
    {
      itLCp[i][j] = CoefficientImageIteratorType(LCparts[i][j], LCparts[i][j]->GetLargestPossibleRegion());
      itLCp[i][j].GoToBegin();
    }
  }

  /** TASK 4A:
   * Do the calculation of the orthonormality subparts.
   *
   ************************************************************************* */

  /** Reset all iterators. */
  it_RCI.GoToBegin();

  if (this->m_CalculateOrthonormalityCondition)
  {
    ScalarType mu1_A, mu2_A, mu3_A, mu1_B, mu2_B, mu3_B, mu1_C, mu2_C, mu3_C;
    ScalarType valueOC;
    while (!itOCp[0][0].IsAtEnd())
    {
      /** Copy values: this way we avoid calling Get() so many times.
       * It also improves code readability.
       */
      mu1_A = itA[0].Get();
      mu2_A = itA[1].Get();
      mu1_B = itB[0].Get();
      mu2_B = itB[1].Get();
      if (ImageDimension == 3)
      {
        mu3_A = itA[2].Get();
        mu3_B = itB[2].Get();
        mu1_C = itC[0].Get();
        mu2_C = itC[1].Get();
        mu3_C = itC[2].Get();
      }
      if (ImageDimension == 2)
      {
        /** Calculate the value of the orthonormality condition. */
        this->m_OrthonormalityConditionValue +=
          it_RCI.Get() * (std::pow(+(1.0 + mu1_A) * (1.0 + mu1_A) + mu2_A * mu2_A - 1.0, 2.0) +
                          std::pow(+mu1_B * mu1_B + (1.0 + mu2_B) * (1.0 + mu2_B) - 1.0, 2.0) +
                          std::pow(+(1.0 + mu1_A) * mu1_B + mu2_A * (1.0 + mu2_B), 2.0));
        /** Calculate the derivative of the orthonormality condition. */
        /** mu1, part 1 */
        valueOC = +2.0 * (1.0 + mu1_A) * (1.0 + mu1_A) * (1.0 + mu1_A) + 2.0 * mu2_A * mu2_A * (1.0 + mu1_A) -
                  2.0 * (1.0 + mu1_A) + mu1_B * mu1_B * (1.0 + mu1_A) + mu2_A * (1.0 + mu2_B) * mu1_B;
        itOCp[0][0].Set(2.0 * valueOC);
        /** mu1, part2*/
        valueOC = +mu1_B * (1.0 + mu1_A) * (1.0 + mu1_A) + mu2_A * (1.0 + mu2_B) * (1.0 + mu1_A) +
                  2.0 * mu1_B * mu1_B * mu1_B + 2.0 * mu1_B * (1.0 + mu2_B) * (1.0 + mu2_B) - 2.0 * mu1_B;
        itOCp[0][1].Set(2.0 * valueOC);
        /** mu2, part 1 */
        valueOC = +2.0 * mu2_A * mu2_A * mu2_A + 2.0 * mu2_A * (1.0 + mu1_A) * (1.0 + mu1_A) - 2.0 * mu2_A +
                  mu2_A * (1.0 + mu2_B) * (1.0 + mu2_B) + mu1_B * (1.0 + mu1_A) * (1.0 + mu2_B);
        itOCp[1][0].Set(2.0 * valueOC);
        /** mu2, part2*/
        valueOC = +mu2_A * mu2_A * (1.0 + mu2_B) + mu1_B * (1.0 + mu1_A) * mu2_A +
                  2.0 * (1.0 + mu2_B) * (1.0 + mu2_B) * (1.0 + mu2_B) + 2.0 * mu1_B * mu1_B * (1.0 + mu2_B) -
                  2.0 * (1.0 + mu2_B);
        itOCp[1][1].Set(2.0 * valueOC);
      } // end if dim == 2
      else if (ImageDimension == 3)
      {
        /** Calculate the value of the orthonormality condition. */
        this->m_OrthonormalityConditionValue +=
          it_RCI.Get() * (std::pow(+(1.0 + mu1_A) * (1.0 + mu1_A) + mu2_A * mu2_A + mu3_A * mu3_A - 1.0, 2.0) +
                          std::pow(+(1.0 + mu1_A) * mu1_B + mu2_A * (1.0 + mu2_B) + mu3_A * mu3_B, 2.0) +
                          std::pow(+(1.0 + mu1_A) * mu1_C + mu2_A * mu2_C + mu3_A * (1.0 + mu3_C), 2.0) +
                          std::pow(+mu1_B * mu1_B + (1.0 + mu2_B) * (1.0 + mu2_B) + mu3_B * mu3_B - 1.0, 2.0) +
                          std::pow(+mu1_B * mu1_C + (1.0 + mu2_B) * mu2_C + mu3_B * (1.0 + mu3_C), 2.0) +
                          std::pow(+mu1_C * mu1_C + mu2_C * mu2_C + (1.0 + mu3_C) * (1.0 + mu3_C) - 1.0, 2.0));
        /** Calculate the derivative of the orthonormality condition. */
        /** mu1, part 1 */
        valueOC = +2.0 * (1.0 + mu1_A) * (1.0 + mu1_A) * (1.0 + mu1_A) + 2.0 * mu2_A * mu2_A * (1.0 + mu1_A) +
                  2.0 * (1.0 + mu1_A) * mu3_A * mu3_A - 2.0 * (1.0 + mu1_A) + mu1_B * mu1_B * (1.0 + mu1_A) +
                  mu2_A * (1.0 + mu2_B) * mu1_B + mu1_B * mu3_A * mu3_B + (1.0 + mu1_A) * mu1_C * mu1_C +
                  mu1_C * mu2_A * mu2_C + mu1_C * mu3_A * (1.0 + mu3_C);
        itOCp[0][0].Set(2.0 * valueOC);
        /** mu1, part2 */
        valueOC = +(1.0 + mu1_A) * (1.0 + mu1_A) * mu1_B + (1.0 + mu1_A) * mu2_A * mu3_B +
                  (1.0 + mu1_A) * mu3_A * mu3_B + mu1_B * mu1_B * mu1_B + mu1_B * (1.0 + mu2_B) * (1.0 + mu2_B) +
                  mu1_B * mu3_B * mu3_B - mu1_B + mu1_B * mu1_C * mu1_C + mu1_C * (1.0 + mu2_B) * mu2_C +
                  mu1_C * mu3_B * (1.0 + mu3_C);
        itOCp[0][1].Set(2.0 * valueOC);
        /** mu1, part3 */
        valueOC = +(1.0 + mu1_A) * (1.0 + mu1_A) * mu1_C + (1.0 + mu1_A) * mu2_A * mu2_C +
                  (1.0 + mu1_A) * mu3_A * (1.0 + mu3_C) + mu1_B * mu1_B * mu1_C + mu1_B * (1.0 + mu2_B) * mu2_C +
                  mu1_B * mu3_B * (1.0 + mu3_C) + 2.0 * mu1_C * mu1_C * mu1_C + 2.0 * mu1_C * mu2_C * mu2_C +
                  2.0 * mu1_C * (1.0 + mu3_C) * (1.0 + mu3_C) - 2.0 * mu1_C;
        itOCp[0][2].Set(2.0 * valueOC);
        /** mu2, part 1 */
        valueOC = +2.0 * mu2_A * mu2_A * mu2_A + 2.0 * mu2_A * (1.0 + mu1_A) * (1.0 + mu1_A) - 2.0 * mu2_A +
                  2.0 * mu2_A * mu3_A * mu3_A + mu2_A * (1.0 + mu2_B) * (1.0 + mu2_B) +
                  mu1_B * (1.0 + mu1_A) * (1.0 + mu2_B) + (1.0 + mu2_B) * mu3_A * mu3_B + mu2_A * mu2_C * mu2_C +
                  (1.0 + mu1_A) * mu1_C * mu2_C + mu2_C * mu3_A * (1.0 + mu3_C);
        itOCp[1][0].Set(2.0 * valueOC);
        /** mu2, part2 */
        valueOC = +mu2_A * mu2_A * (1.0 + mu2_B) + mu1_B * (1.0 + mu1_A) * mu2_A + mu2_A * mu3_A * mu3_B +
                  2.0 * (1.0 + mu2_B) * (1.0 + mu2_B) * (1.0 + mu2_B) + 2.0 * mu1_B * mu1_B * (1.0 + mu2_B) -
                  2.0 * (1.0 + mu2_B) + 2.0 * (1.0 + mu2_B) * mu3_B * mu3_B + (1.0 + mu2_B) * mu2_C * mu2_C +
                  mu1_B * mu1_C * mu2_C + mu2_C * mu3_B * (1.0 + mu3_C);
        itOCp[1][1].Set(2.0 * valueOC);
        /** mu2, part 3 */
        valueOC = +mu2_A * mu2_A * mu2_C + (1.0 + mu1_A) * mu1_C * mu2_A + mu2_A * mu3_A * (1.0 + mu3_C) +
                  (1.0 + mu2_B) * (1.0 + mu2_B) * mu2_C + mu1_B * mu1_C * mu2_B +
                  (1.0 + mu2_B) * mu3_B * (1.0 + mu3_C) + 2.0 * mu2_C * mu2_C * mu2_C + 2.0 * mu1_C * mu1_C * mu2_C +
                  2.0 * mu2_C * (1.0 + mu3_C) * (1.0 + mu3_C) - 2.0 * mu2_C;
        itOCp[1][2].Set(2.0 * valueOC);
        /** mu3, part 1 */
        valueOC = +2.0 * mu3_A * mu3_A * mu3_A + 2.0 * mu3_A * (1.0 + mu1_A) * (1.0 + mu1_A) - 2.0 * mu3_A +
                  2.0 * mu2_A * mu2_A * mu3_A + mu3_A * mu3_B * mu3_B + mu1_B * (1.0 + mu1_A) * mu3_B +
                  (1.0 + mu2_B) * mu2_A * mu3_B + mu3_A * (1.0 + mu3_C) * (1.0 + mu3_C) +
                  (1.0 + mu1_A) * mu1_C * (1.0 + mu3_C) + mu2_C * mu2_A * (1.0 + mu3_C);
        itOCp[2][0].Set(2.0 * valueOC);
        /** mu3, part2 */
        valueOC = +mu3_A * mu3_A * mu3_B + mu1_B * (1.0 + mu1_A) * mu3_A + mu2_A * mu3_A * (1.0 + mu2_B) +
                  2.0 * mu3_B * mu3_B * mu3_B + 2.0 * mu1_B * mu1_B * mu3_B - 2.0 * mu3_B +
                  2.0 * (1.0 + mu2_B) * (1.0 + mu2_B) * mu3_B + mu3_B * (1.0 + mu3_C) * (1.0 + mu3_C) +
                  mu1_B * mu1_C * (1.0 + mu3_C) + mu2_C * (1.0 + mu2_B) * (1.0 + mu3_C);
        itOCp[2][1].Set(2.0 * valueOC);
        /** mu3, part 3 */
        valueOC = +mu3_A * mu3_A * (1.0 + mu3_C) + (1.0 + mu1_A) * mu1_C * mu3_A + mu2_A * mu3_A * mu2_C +
                  mu3_B * mu3_B * (1.0 + mu3_C) + mu1_B * mu1_C * mu3_B + (1.0 + mu2_B) * mu3_B * mu2_C +
                  2.0 * (1.0 + mu3_C) * (1.0 + mu3_C) * (1.0 + mu3_C) + 2.0 * mu1_C * mu1_C * (1.0 + mu3_C) +
                  2.0 * mu2_C * mu2_C * (1.0 + mu3_C) - 2.0 * (1.0 + mu3_C);
        itOCp[2][2].Set(2.0 * valueOC);
      } // end if dim == 3

      /** Increase all iterators. */
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        ++itA[i];
        ++itB[i];
        if (ImageDimension == 3)
        {
          ++itC[i];
        }
        for (unsigned int j = 0; j < ImageDimension; ++j)
        {
          ++itOCp[i][j];
        }
      }
      ++it_RCI;

    } // end while
  }   // end if do orthonormality

  /** TASK 4B:
   * Do the calculation of the properness parts.
   *
   ************************************************************************* */

  /** Reset all iterators. */
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    itA[i].GoToBegin();
    itB[i].GoToBegin();
    if (ImageDimension == 3)
    {
      itC[i].GoToBegin();
    }
  }
  it_RCI.GoToBegin();

  if (this->m_CalculatePropernessCondition)
  {
    ScalarType mu1_A, mu2_A, mu3_A, mu1_B, mu2_B, mu3_B, mu1_C, mu2_C, mu3_C;
    ScalarType valuePC;
    while (!itPCp[0][0].IsAtEnd())
    {
      /** Copy values: this way we avoid calling Get() so many times.
       * It also improves code readability.
       */
      mu1_A = itA[0].Get();
      mu2_A = itA[1].Get();
      mu1_B = itB[0].Get();
      mu2_B = itB[1].Get();
      if (ImageDimension == 3)
      {
        mu3_A = itA[2].Get();
        mu3_B = itB[2].Get();
        mu1_C = itC[0].Get();
        mu2_C = itC[1].Get();
        mu3_C = itC[2].Get();
      }
      if (ImageDimension == 2)
      {
        /** Calculate the value of the properness condition. */
        this->m_PropernessConditionValue +=
          it_RCI.Get() * (std::pow(+(1.0 + mu1_A) * (1.0 + mu2_B) - mu2_A * mu1_B - 1.0, 2.0));
        /** Calculate the derivative of the properness condition. */
        /** mu1, part 1 */
        valuePC = +(1.0 + mu2_B) * (1.0 + mu2_B) * (1.0 + mu1_A) - mu2_A * (1.0 + mu2_B) * mu1_B - (1.0 + mu2_B);
        itPCp[0][0].Set(2.0 * valuePC);
        /** mu1, part 2 */
        valuePC = +mu2_A + mu2_A * mu2_A * mu1_B - mu2_A * (1.0 + mu2_B) * (1.0 + mu1_A);
        itPCp[0][1].Set(2.0 * valuePC);
        /** mu2, part 1 */
        valuePC = +mu1_B * mu1_B * mu2_A - mu1_B * (1.0 + mu1_A) * (1.0 + mu2_B) + mu1_B;
        itPCp[1][0].Set(2.0 * valuePC);
        /** mu2, part 2 */
        valuePC = -(1.0 + mu1_A) + (1.0 + mu1_A) * (1.0 + mu1_A) * (1.0 + mu2_B) - mu1_B * (1.0 + mu1_A) * mu2_A;
        itPCp[1][1].Set(2.0 * valuePC);
      } // end if dim == 2
      else if (ImageDimension == 3)
      {
        /** Calculate the value of the properness condition. */
        this->m_PropernessConditionValue +=
          it_RCI.Get() * (std::pow(-mu1_C * (1.0 + mu2_B) * mu3_A + mu1_B * mu2_C * mu3_A + mu1_C * mu2_A * mu3_B -
                                     (1.0 + mu1_A) * mu2_C * mu3_B - mu1_B * mu2_A * (1.0 + mu3_C) +
                                     (1.0 + mu1_A) * (1.0 + mu2_B) * (1.0 + mu3_C) - 1.0,
                                   2.0));
        /** Calculate the derivative of the properness condition. */
        /** mu1, part 1 */
        valuePC = +(1.0 + mu1_A) * mu2_C * mu2_C * mu3_B * mu3_B +
                  (1.0 + mu1_A) * (1.0 + mu2_B) * (1.0 + mu2_B) * (1.0 + mu3_C) * (1.0 + mu3_C) +
                  mu1_C * (1.0 + mu2_B) * mu2_C * mu3_A * mu3_B -
                  mu1_C * (1.0 + mu2_B) * (1.0 + mu2_B) * mu3_A * (1.0 + mu3_C) -
                  mu1_B * mu2_C * mu2_C * mu3_A * mu3_B + mu1_B * (1.0 + mu2_B) * mu2_C * mu3_A * (1.0 + mu3_C) -
                  mu1_C * mu2_A * mu2_C * mu3_B * mu3_B + mu1_C * mu2_A * (1.0 + mu2_B) * mu3_B * (1.0 + mu3_C) +
                  mu1_B * mu2_A * mu2_C * mu3_B * (1.0 + mu3_C) -
                  2.0 * (1.0 + mu1_A) * (1.0 + mu2_B) * mu2_C * mu3_B * (1.0 + mu3_C) + mu2_C * mu3_B -
                  mu1_B * mu2_A * (1.0 + mu2_B) * (1.0 + mu3_C) * (1.0 + mu3_C) - (1.0 + mu2_B) * (1.0 + mu3_C);
        itPCp[0][0].Set(2.0 * valuePC);
        /** mu1, part 2 */
        valuePC = +mu1_B * mu2_C * mu2_C * mu3_A * mu3_A + mu1_B * mu2_A * mu2_A * (1.0 + mu3_C) * (1.0 + mu3_C) -
                  mu1_C * (1.0 + mu2_B) * mu2_C * mu3_A * mu3_A +
                  mu1_C * mu2_A * (1.0 + mu2_B) * mu3_A * (1.0 + mu3_C) + mu1_C * mu2_A * mu2_C * mu3_A * mu3_B -
                  (1.0 + mu1_A) * mu2_C * mu2_C * mu3_A * mu3_B - 2.0 * mu1_B * mu2_A * mu2_C * mu3_A * (1.0 + mu3_C) +
                  (1.0 + mu1_A) * (1.0 + mu2_B) * mu2_C * mu3_A * (1.0 + mu3_C) - mu2_C * mu3_A -
                  mu1_C * mu2_A * mu2_A * mu3_B * (1.0 + mu3_C) +
                  (1.0 + mu1_A) * mu2_A * mu2_C * mu3_B * (1.0 + mu3_C) -
                  (1.0 + mu1_A) * mu2_A * (1.0 + mu2_B) * (1.0 + mu3_C) * (1.0 + mu3_C) + mu2_A * (1.0 + mu3_C);
        itPCp[0][1].Set(2.0 * valuePC);
        /** mu1, part 3 */
        valuePC = +mu1_C * (1.0 + mu2_B) * (1.0 + mu2_B) * mu3_A * mu3_A + mu1_C * mu2_A * mu2_A * mu3_B * mu3_B -
                  mu1_B * (1.0 + mu2_B) * mu2_C * mu3_A * mu3_A - 2.0 * mu1_C * mu2_A * (1.0 + mu2_B) * mu3_A * mu3_B +
                  (1.0 + mu1_A) * (1.0 + mu2_B) * mu2_C * mu3_A * mu3_B +
                  mu1_B * mu2_A * (1.0 + mu2_B) * mu3_A * (1.0 + mu3_C) -
                  (1.0 + mu1_A) * (1.0 + mu2_B) * (1.0 + mu2_B) * mu3_A * (1.0 + mu3_C) + (1.0 + mu2_B) * mu3_A +
                  mu1_B * mu2_A * mu2_C * mu3_A * mu3_B - (1.0 + mu1_A) * mu2_A * mu2_C * mu3_B * mu3_B -
                  mu1_B * mu2_A * mu2_A * mu3_B * (1.0 + mu3_C) +
                  (1.0 + mu1_A) * mu2_A * (1.0 + mu2_B) * mu3_B * (1.0 + mu3_C) - mu2_A * mu3_B;
        itPCp[0][2].Set(2.0 * valuePC);
        /** mu2, part 1 */
        valuePC = +mu1_C * mu1_C * mu2_A * mu3_B * mu3_B + mu1_B * mu1_B * mu2_A * (1.0 + mu3_C) * (1.0 + mu3_C) -
                  mu1_C * mu1_C * (1.0 + mu2_B) * mu3_A * mu3_B +
                  mu1_B * mu1_C * (1.0 + mu2_B) * mu3_A * (1.0 + mu3_C) + mu1_B * mu1_C * mu2_C * mu3_A * mu3_B -
                  mu1_B * mu1_B * mu2_C * mu3_A * (1.0 + mu3_C) - (1.0 + mu1_A) * mu1_C * mu2_C * mu3_B * mu3_B -
                  2.0 * mu1_B * mu1_C * mu2_A * mu3_B * (1.0 + mu3_C) +
                  (1.0 + mu1_A) * mu1_C * (1.0 + mu2_B) * mu3_B * (1.0 + mu3_C) - mu1_C * mu3_B +
                  (1.0 + mu1_A) * mu1_B * mu2_C * mu3_B * (1.0 + mu3_C) -
                  (1.0 + mu1_A) * mu1_B * (1.0 + mu2_B) * (1.0 + mu3_C) * (1.0 + mu3_C) + mu1_B * (1.0 + mu3_C);
        itPCp[1][0].Set(2.0 * valuePC);
        /** mu2, part 2 */
        valuePC = +mu1_C * mu1_C * (1.0 + mu2_B) * mu3_A * mu3_A +
                  (1.0 + mu1_A) * (1.0 + mu1_A) * (1.0 + mu2_B) * (1.0 + mu3_C) * (1.0 + mu3_C) -
                  mu1_B * mu1_C * mu2_C * mu3_A * mu3_A - mu1_C * mu1_C * mu2_A * mu3_A * mu3_B +
                  (1.0 + mu1_A) * mu1_C * mu2_C * mu3_A * mu3_B + mu1_B * mu1_C * mu2_A * mu3_A * (1.0 + mu3_C) -
                  2.0 * (1.0 + mu1_A) * mu1_C * (1.0 + mu2_B) * mu3_A * (1.0 + mu3_C) + mu1_C * mu3_A +
                  (1.0 + mu1_A) * mu1_B * mu2_C * mu3_A * (1.0 + mu3_C) +
                  (1.0 + mu1_A) * mu1_C * mu2_A * mu3_B * (1.0 + mu3_C) -
                  (1.0 + mu1_A) * (1.0 + mu1_A) * mu2_C * mu3_B * (1.0 + mu3_C) -
                  (1.0 + mu1_A) * mu1_B * mu2_A * (1.0 + mu3_C) * (1.0 + mu3_C) - (1.0 + mu1_A) * (1.0 + mu3_C);
        itPCp[1][1].Set(2.0 * valuePC);
        /** mu2, part 3 */
        valuePC = +mu1_B * mu1_B * mu2_C * mu3_A * mu3_A + (1.0 + mu1_A) * (1.0 + mu1_A) * mu2_C * mu3_B * mu3_B -
                  mu1_B * mu1_C * (1.0 + mu2_B) * mu3_A * mu3_A +
                  (1.0 + mu1_A) * mu1_C * (1.0 + mu2_B) * mu3_A * mu3_B + mu1_B * mu1_C * mu2_A * mu3_A * mu3_B -
                  2.0 * (1.0 + mu1_A) * mu1_B * mu2_C * mu3_A * mu3_B - mu1_B * mu1_B * mu2_A * mu3_A * (1.0 + mu3_C) +
                  (1.0 + mu1_A) * mu1_B * (1.0 + mu2_B) * mu3_A * (1.0 + mu3_C) - mu1_B * mu3_A -
                  (1.0 + mu1_A) * mu1_C * mu2_A * mu3_B * mu3_B +
                  (1.0 + mu1_A) * mu1_B * mu2_A * mu3_B * (1.0 + mu3_C) -
                  (1.0 + mu1_A) * (1.0 + mu1_A) * (1.0 + mu2_B) * mu3_B * (1.0 + mu3_C) + (1.0 + mu1_A) * mu3_B;
        itPCp[1][2].Set(2.0 * valuePC);
        /** mu3, part 1 */
        valuePC = +mu1_C * mu1_C * (1.0 + mu2_B) * (1.0 + mu2_B) * mu3_A + mu1_B * mu1_B * mu2_C * mu2_C * mu3_A -
                  2.0 * mu1_B * mu1_C * (1.0 + mu2_B) * mu2_C * mu3_A - mu1_C * mu1_C * mu2_A * (1.0 + mu2_B) * mu3_B +
                  (1.0 + mu1_A) * mu1_C * (1.0 + mu2_B) * mu2_C * mu3_B +
                  mu1_B * mu1_C * mu2_A * (1.0 + mu2_B) * (1.0 + mu3_C) -
                  (1.0 + mu1_A) * mu1_C * (1.0 + mu2_B) * (1.0 + mu2_B) * (1.0 + mu3_C) + mu1_C * (1.0 + mu2_B) +
                  mu1_B * mu1_C * mu2_A * mu2_C * mu3_B - (1.0 + mu1_A) * mu1_B * mu2_C * mu2_C * mu3_B -
                  mu1_B * mu1_B * mu2_A * mu2_C * (1.0 + mu3_C) +
                  (1.0 + mu1_A) * mu1_B * (1.0 + mu2_B) * mu2_C * (1.0 + mu3_C) + mu1_B * mu2_C;
        itPCp[2][0].Set(2.0 * valuePC);
        /** mu3, part 2 */
        valuePC = +mu1_C * mu1_C * mu2_A * mu2_A * mu3_B + (1.0 + mu1_A) * (1.0 + mu1_A) * mu2_C * mu2_C * mu3_B -
                  mu1_C * mu1_C * mu2_A * (1.0 + mu2_B) * mu3_A +
                  (1.0 + mu1_A) * mu1_C * (1.0 + mu2_B) * mu2_C * mu3_A + mu1_B * mu1_C * mu2_A * mu2_C * mu3_A -
                  (1.0 + mu1_A) * mu1_B * mu2_C * mu2_C * mu3_A - 2.0 * (1.0 + mu1_A) * mu1_C * mu2_A * mu2_C * mu3_B -
                  mu1_B * mu1_C * mu2_A * mu2_A * (1.0 + mu3_C) +
                  (1.0 + mu1_A) * mu1_C * mu2_A * (1.0 + mu2_B) * (1.0 + mu3_C) - mu1_C * mu2_A +
                  (1.0 + mu1_A) * mu1_B * mu2_A * mu2_C * (1.0 + mu3_C) -
                  (1.0 + mu1_A) * (1.0 + mu1_A) * (1.0 + mu2_B) * mu2_C * (1.0 + mu3_C) + (1.0 + mu1_A) * mu2_C;
        itPCp[2][1].Set(2.0 * valuePC);
        /** mu3, part 3 */
        valuePC = +mu1_B * mu1_B * mu2_A * mu2_A * (1.0 + mu3_C) +
                  (1.0 + mu1_A) * (1.0 + mu1_A) * (1.0 + mu2_B) * (1.0 + mu2_B) * (1.0 + mu3_C) +
                  mu1_B * mu1_C * mu2_A * (1.0 + mu2_B) * mu3_A -
                  (1.0 + mu1_A) * mu1_C * (1.0 + mu2_B) * (1.0 + mu2_B) * mu3_A -
                  mu1_B * mu1_B * mu2_A * mu2_C * mu3_A + (1.0 + mu1_A) * mu1_B * (1.0 + mu2_B) * mu2_C * mu3_A -
                  mu1_B * mu1_C * mu2_A * mu2_A * mu3_B + (1.0 + mu1_A) * mu1_C * mu2_A * (1.0 + mu2_B) * mu3_B +
                  (1.0 + mu1_A) * mu1_B * mu2_A * mu2_C * mu3_B +
                  (1.0 + mu1_A) * (1.0 + mu1_A) * (1.0 + mu2_B) * mu2_C * mu3_B -
                  2.0 * (1.0 + mu1_A) * mu1_B * mu2_A * (1.0 + mu2_B) * (1.0 + mu3_C) + mu1_B * mu2_A -
                  (1.0 + mu1_A) * (1.0 + mu2_B);
        itPCp[2][2].Set(2.0 * valuePC);
      } // end if dim == 3

      /** Increase all iterators. */
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        ++itA[i];
        ++itB[i];
        if (ImageDimension == 3)
        {
          ++itC[i];
        }
        for (unsigned int j = 0; j < ImageDimension; ++j)
        {
          ++itPCp[i][j];
        }
      }
      ++it_RCI;

    } // end while
  }   // end if do properness

  /** TASK 4C:
   * Do the calculation of the linearity parts.
   *
   ************************************************************************* */

  /** Reset all iterators. */
  it_RCI.GoToBegin();

  if (this->m_CalculateLinearityCondition)
  {
    while (!itLCp[0][0].IsAtEnd())
    {
      /** Linearity condition part. */
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        /** Calculate the value of the linearity condition. */
        this->m_LinearityConditionValue +=
          it_RCI.Get() * (+itD[i].Get() * itD[i].Get() + itE[i].Get() * itE[i].Get() + itG[i].Get() * itG[i].Get());
        if (ImageDimension == 3)
        {
          this->m_LinearityConditionValue +=
            it_RCI.Get() * (+itF[i].Get() * itF[i].Get() + itH[i].Get() * itH[i].Get() + itI[i].Get() * itI[i].Get());
        }
      } // end loop over i

      /** Calculate the derivative of the linearity condition. */
      if (ImageDimension == 2)
      {
        itLCp[0][0].Set(2.0 * itD[0].Get());
        itLCp[0][1].Set(2.0 * itE[0].Get());
        itLCp[0][2].Set(2.0 * itG[0].Get());
        itLCp[1][0].Set(2.0 * itD[1].Get());
        itLCp[1][1].Set(2.0 * itE[1].Get());
        itLCp[1][2].Set(2.0 * itG[1].Get());
      } // end if dim == 2
      else if (ImageDimension == 3)
      {
        itLCp[0][0].Set(2.0 * itD[0].Get());
        itLCp[0][1].Set(2.0 * itE[0].Get());
        itLCp[0][2].Set(2.0 * itG[0].Get());
        itLCp[0][3].Set(2.0 * itF[0].Get());
        itLCp[0][4].Set(2.0 * itH[0].Get());
        itLCp[0][5].Set(2.0 * itI[0].Get());
        itLCp[1][0].Set(2.0 * itD[1].Get());
        itLCp[1][1].Set(2.0 * itE[1].Get());
        itLCp[1][2].Set(2.0 * itG[1].Get());
        itLCp[1][3].Set(2.0 * itF[1].Get());
        itLCp[1][4].Set(2.0 * itH[1].Get());
        itLCp[1][5].Set(2.0 * itI[1].Get());
        itLCp[2][0].Set(2.0 * itD[2].Get());
        itLCp[2][1].Set(2.0 * itE[2].Get());
        itLCp[2][2].Set(2.0 * itG[2].Get());
        itLCp[2][3].Set(2.0 * itF[2].Get());
        itLCp[2][4].Set(2.0 * itH[2].Get());
        itLCp[2][5].Set(2.0 * itI[2].Get());
      } // end if dim == 3

      /** Increase all iterators. */
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        ++itD[i];
        ++itE[i];
        ++itG[i];
        if (ImageDimension == 3)
        {
          ++itF[i];
          ++itH[i];
          ++itI[i];
        }
        for (unsigned int j = 0; j < NofLParts; ++j)
        {
          ++itLCp[i][j];
        }
      }
      ++it_RCI;

    } // end while
  }   // end if do linearity

  /** TASK 5:
   * Do the actual calculation of the rigidity penalty term value.
   *
   ************************************************************************* */

  /** Calculate the rigidity penalty term value. */
  if (this->m_CalculateLinearityCondition)
  {
    this->m_LinearityConditionValue /= rigidityCoefficientSum;
  }
  if (this->m_CalculateOrthonormalityCondition)
  {
    this->m_OrthonormalityConditionValue /= rigidityCoefficientSum;
  }
  if (this->m_CalculatePropernessCondition)
  {
    this->m_PropernessConditionValue /= rigidityCoefficientSum;
  }

  if (this->m_UseLinearityCondition)
  {
    this->m_RigidityPenaltyTermValue += this->m_LinearityConditionWeight * this->m_LinearityConditionValue;
  }
  if (this->m_UseOrthonormalityCondition)
  {
    this->m_RigidityPenaltyTermValue += this->m_OrthonormalityConditionWeight * this->m_OrthonormalityConditionValue;
  }
  if (this->m_UsePropernessCondition)
  {
    this->m_RigidityPenaltyTermValue += this->m_PropernessConditionWeight * this->m_PropernessConditionValue;
  }
  value = this->m_RigidityPenaltyTermValue;

  /** TASK 6:
   * Create filtered versions of the subparts.
   * Create all necessary iterators and operators.
   ************************************************************************* */

  /** Create filtered orthonormality, properness and linearity parts. */
  std::vector<CoefficientImagePointer> OCpartsF(ImageDimension);
  std::vector<CoefficientImagePointer> PCpartsF(ImageDimension);
  std::vector<CoefficientImagePointer> LCpartsF(ImageDimension);
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    OCpartsF[i] = CoefficientImageType::New();
    OCpartsF[i]->SetRegions(inputImages[0]->GetLargestPossibleRegion());
    OCpartsF[i]->Allocate();
    PCpartsF[i] = CoefficientImageType::New();
    PCpartsF[i]->SetRegions(inputImages[0]->GetLargestPossibleRegion());
    PCpartsF[i]->Allocate();
    LCpartsF[i] = CoefficientImageType::New();
    LCpartsF[i]->SetRegions(inputImages[0]->GetLargestPossibleRegion());
    LCpartsF[i]->Allocate();
  }

  /** Create neighborhood iterators over the subparts. */
  std::vector<std::vector<NeighborhoodIteratorType>> nitOCp(ImageDimension);
  std::vector<std::vector<NeighborhoodIteratorType>> nitPCp(ImageDimension);
  std::vector<std::vector<NeighborhoodIteratorType>> nitLCp(ImageDimension);
  RadiusType                                         radius;
  radius.Fill(1);
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    nitOCp[i].resize(ImageDimension);
    nitPCp[i].resize(ImageDimension);
    nitLCp[i].resize(NofLParts);
    for (unsigned int j = 0; j < ImageDimension; ++j)
    {
      nitOCp[i][j] = NeighborhoodIteratorType(radius, OCparts[i][j], OCparts[i][j]->GetLargestPossibleRegion());
      nitOCp[i][j].GoToBegin();
      nitPCp[i][j] = NeighborhoodIteratorType(radius, PCparts[i][j], PCparts[i][j]->GetLargestPossibleRegion());
      nitPCp[i][j].GoToBegin();
    }
    for (unsigned int j = 0; j < NofLParts; ++j)
    {
      nitLCp[i][j] = NeighborhoodIteratorType(radius, LCparts[i][j], LCparts[i][j]->GetLargestPossibleRegion());
      nitLCp[i][j].GoToBegin();
    }
  }

  /** Create iterators over the filtered parts. */
  std::vector<CoefficientImageIteratorType> itOCpf(ImageDimension);
  std::vector<CoefficientImageIteratorType> itPCpf(ImageDimension);
  std::vector<CoefficientImageIteratorType> itLCpf(ImageDimension);
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    itOCpf[i] = CoefficientImageIteratorType(OCpartsF[i], OCpartsF[i]->GetLargestPossibleRegion());
    itOCpf[i].GoToBegin();
    itPCpf[i] = CoefficientImageIteratorType(PCpartsF[i], PCpartsF[i]->GetLargestPossibleRegion());
    itPCpf[i].GoToBegin();
    itLCpf[i] = CoefficientImageIteratorType(LCpartsF[i], LCpartsF[i]->GetLargestPossibleRegion());
    itLCpf[i].GoToBegin();
  }

  /** Create a neigborhood iterator over the rigidity image. */
  NeighborhoodIteratorType nit_RCI(
    radius, this->m_RigidityCoefficientImage, this->m_RigidityCoefficientImage->GetLargestPossibleRegion());
  nit_RCI.GoToBegin();
  unsigned int neighborhoodSize = nit_RCI.Size();

  /** Create ND operators. */
  NeighborhoodType Operator_A, Operator_B, Operator_C, Operator_D, Operator_E, Operator_F, Operator_G, Operator_H,
    Operator_I;
  this->CreateNDOperator(Operator_A, "FA", spacing);
  this->CreateNDOperator(Operator_B, "FB", spacing);
  if (ImageDimension == 3)
  {
    this->CreateNDOperator(Operator_C, "FC", spacing);
  }

  if (this->m_CalculateLinearityCondition)
  {
    this->CreateNDOperator(Operator_D, "FD", spacing);
    this->CreateNDOperator(Operator_E, "FE", spacing);
    this->CreateNDOperator(Operator_G, "FG", spacing);
    if (ImageDimension == 3)
    {
      this->CreateNDOperator(Operator_F, "FF", spacing);
      this->CreateNDOperator(Operator_H, "FH", spacing);
      this->CreateNDOperator(Operator_I, "FI", spacing);
    }
  }

  /** TASK 7A:
   * Calculate the filtered versions of the orthonormality subparts.
   * These are F_A * {subpart_0} + F_B * {subpart_1},
   * and (for 3D) + F_C * {subpart_2}, for all dimensions.
   ************************************************************************* */

  if (this->m_CalculateOrthonormalityCondition)
  {
    while (!itOCpf[0].IsAtEnd())
    {
      /** Create and reset tmp with zeros. */
      std::vector<double> tmp(ImageDimension, 0.0);

      /** Loop over all dimensions. */
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        /** Loop over the neighborhood. */
        for (unsigned int k = 0; k < neighborhoodSize; ++k)
        {
          /** Calculation of the inner product. */
          tmp[i] += Operator_A.GetElement(k)   // FA *
                    * nitOCp[i][0].GetPixel(k) // subpart[ i ][ 0 ]
                    * nit_RCI.GetPixel(k);     // c(k)
          tmp[i] += Operator_B.GetElement(k)   // FB *
                    * nitOCp[i][1].GetPixel(k) // subpart[ i ][ 1 ]
                    * nit_RCI.GetPixel(k);     // c(k)
          if (ImageDimension == 3)
          {
            tmp[i] += Operator_C.GetElement(k)   // FC *
                      * nitOCp[i][2].GetPixel(k) // subpart[ i ][ 2 ]
                      * nit_RCI.GetPixel(k);     // c(k)
          }
        } // end loop over neighborhood

        /** Set the result in the filtered part. */
        itOCpf[i].Set(tmp[i]);

      } // end loop over dimension i

      /** Increase all iterators. */
      ++nit_RCI;
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        ++itOCpf[i];
        for (unsigned int j = 0; j < ImageDimension; ++j)
        {
          ++nitOCp[i][j];
        }
      }
    } // end while
  }   // end if do orthonormality

  /** TASK 7B:
   * Calculate the filtered versions of the properness subparts.
   * These are F_A * {subpart_0} + F_B * {subpart_1},
   * and (for 3D) + F_C * {subpart_2}, for all dimensions.
   ************************************************************************* */

  nit_RCI.GoToBegin();
  if (this->m_CalculatePropernessCondition)
  {
    while (!itPCpf[0].IsAtEnd())
    {
      /** Create and reset tmp with zeros. */
      std::vector<double> tmp(ImageDimension, 0.0);

      /** Loop over all dimensions. */
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        /** Loop over the neighborhood. */
        for (unsigned int k = 0; k < neighborhoodSize; ++k)
        {
          /** Calculation of the inner product. */
          tmp[i] += Operator_A.GetElement(k)   // FA *
                    * nitPCp[i][0].GetPixel(k) // subpart[ i ][ 0 ]
                    * nit_RCI.GetPixel(k);     // c(k)
          tmp[i] += Operator_B.GetElement(k)   // FB *
                    * nitPCp[i][1].GetPixel(k) // subpart[ i ][ 1 ]
                    * nit_RCI.GetPixel(k);     // c(k)
          if (ImageDimension == 3)
          {
            tmp[i] += Operator_C.GetElement(k)   // FC *
                      * nitPCp[i][2].GetPixel(k) // subpart[ i ][ 2 ]
                      * nit_RCI.GetPixel(k);     // c(k)
          }
        } // end loop over neighborhood

        /** Set the result in the filtered part. */
        itPCpf[i].Set(tmp[i]);

      } // end loop over dimension i

      /** Increase all iterators. */
      ++nit_RCI;
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        ++itPCpf[i];
        for (unsigned int j = 0; j < ImageDimension; ++j)
        {
          ++nitPCp[i][j];
        }
      }
    } // end while
  }   // end if do properness

  /** TASK 7C:
   * Calculate the filtered versions of the linearity subparts.
   * These are sum_{i=1}^{NofLParts} F_{D,E,G,F,H,I} * {subpart_i}.
   ************************************************************************* */

  nit_RCI.GoToBegin();
  if (this->m_CalculateLinearityCondition)
  {
    while (!itLCpf[0].IsAtEnd())
    {
      /** Create and reset tmp with zeros. */
      std::vector<double> tmp(ImageDimension, 0.0);

      /** Loop over all dimensions. */
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        /** Loop over the neighborhood. */
        for (unsigned int k = 0; k < neighborhoodSize; ++k)
        {
          /** Calculation of the inner product. */
          tmp[i] += Operator_D.GetElement(k)   // FD *
                    * nitLCp[i][0].GetPixel(k) // subpart[ i ][ 0 ]
                    * nit_RCI.GetPixel(k);     // c(k)
          tmp[i] += Operator_E.GetElement(k)   // FE *
                    * nitLCp[i][1].GetPixel(k) // subpart[ i ][ 1 ]
                    * nit_RCI.GetPixel(k);     // c(k)
          tmp[i] += Operator_G.GetElement(k)   // FG *
                    * nitLCp[i][2].GetPixel(k) // subpart[ i ][ 1 ]
                    * nit_RCI.GetPixel(k);     // c(k)
          if (ImageDimension == 3)
          {
            tmp[i] += Operator_F.GetElement(k)   // FF *
                      * nitLCp[i][3].GetPixel(k) // subpart[ i ][ 1 ]
                      * nit_RCI.GetPixel(k);     // c(k)
            tmp[i] += Operator_H.GetElement(k)   // FH *
                      * nitLCp[i][4].GetPixel(k) // subpart[ i ][ 1 ]
                      * nit_RCI.GetPixel(k);     // c(k)
            tmp[i] += Operator_I.GetElement(k)   // FI *
                      * nitLCp[i][5].GetPixel(k) // subpart[ i ][ 1 ]
                      * nit_RCI.GetPixel(k);     // c(k)
          }
        } // end loop over neighborhood

        /** Set the result in the filtered part. */
        itLCpf[i].Set(tmp[i]);

      } // end loop over dimension i

      /** Increase all iterators. */
      ++nit_RCI;
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        ++itLCpf[i];
        for (unsigned int j = 0; j < NofLParts; ++j)
        {
          ++nitLCp[i][j];
        }
      }
    } // end while
  }   // end if do linearity

  /** TASK 8:
   * Add it all to create the final derivative images.
   ************************************************************************* */

  /** Create derivative images, each holding a component of the vector field. */
  std::vector<CoefficientImagePointer> derivativeImages(ImageDimension);
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    derivativeImages[i] = CoefficientImageType::New();
    derivativeImages[i]->SetRegions(inputImages[i]->GetLargestPossibleRegion());
    derivativeImages[i]->Allocate();
  }

  /** Create iterators over the derivative images. */
  std::vector<CoefficientImageIteratorType> itDIs(ImageDimension);
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    itDIs[i] = CoefficientImageIteratorType(derivativeImages[i], derivativeImages[i]->GetLargestPossibleRegion());
    itDIs[i].GoToBegin();
    itOCpf[i].GoToBegin();
    itPCpf[i].GoToBegin();
    itLCpf[i].GoToBegin();
  }

  /** Do the addition. */
  // NOTE: unlike the values, for the derivatives weight * derivative is returned.
  MeasureType gradMagLC = NumericTraits<MeasureType>::Zero;
  MeasureType gradMagOC = NumericTraits<MeasureType>::Zero;
  MeasureType gradMagPC = NumericTraits<MeasureType>::Zero;
  double      rigidityCoefficientSumSqr = rigidityCoefficientSum * rigidityCoefficientSum;
  while (!itDIs[0].IsAtEnd())
  {
    for (unsigned int i = 0; i < ImageDimension; ++i)
    {
      ScalarType tmpDIs = NumericTraits<ScalarType>::Zero;

      /** Compute gradient magnitude of LC. */
      ScalarType tmpLC = this->m_LinearityConditionWeight * itLCpf[i].Get();
      gradMagLC += tmpLC * tmpLC / rigidityCoefficientSumSqr;

      /** Compute gradient magnitude of OC. */
      ScalarType tmpOC = this->m_OrthonormalityConditionWeight * itOCpf[i].Get();
      gradMagOC += tmpOC * tmpOC / rigidityCoefficientSumSqr;

      /** Compute gradient magnitude of PC. */
      ScalarType tmpPC = this->m_PropernessConditionWeight * itPCpf[i].Get();
      gradMagPC += tmpPC * tmpPC / rigidityCoefficientSumSqr;

      /** Compute derivative contribution. */
      if (this->m_UseLinearityCondition)
      {
        tmpDIs += tmpLC;
      }
      if (this->m_UseOrthonormalityCondition)
      {
        tmpDIs += tmpOC;
      }
      if (this->m_UsePropernessCondition)
      {
        tmpDIs += tmpPC;
      }
      itDIs[i].Set(tmpDIs);

      /** Update iterators. */
      ++itDIs[i];
      ++itOCpf[i];
      ++itPCpf[i];
      ++itLCpf[i];
    }
  } // end while

  /** Set the gradient magnitudes of the several terms. */
  this->m_LinearityConditionGradientMagnitude = std::sqrt(gradMagLC);
  this->m_OrthonormalityConditionGradientMagnitude = std::sqrt(gradMagOC);
  this->m_PropernessConditionGradientMagnitude = std::sqrt(gradMagPC);

  /** Rearrange to create a derivative. */
  unsigned int j = 0;
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    itDIs[i].GoToBegin();
    while (!itDIs[i].IsAtEnd())
    {
      derivative[j] = itDIs[i].Get() / rigidityCoefficientSum;
      ++itDIs[i];
      ++j;
    } // end while
  }   // end for

} // end GetValueAndDerivative()


/**
 * ********************* PrintSelf ******************************
 */

template <class TFixedImage, class TScalarType>
void
TransformRigidityPenaltyTerm<TFixedImage, TScalarType>::PrintSelf(std::ostream & os, Indent indent) const
{
  /** Call the superclass' PrintSelf. */
  Superclass::PrintSelf(os, indent);

  /** Add debugging information. */
  os << indent << "LinearityConditionWeight: " << this->m_LinearityConditionWeight << std::endl;
  os << indent << "OrthonormalityConditionWeight: " << this->m_OrthonormalityConditionWeight << std::endl;
  os << indent << "PropernessConditionWeight: " << this->m_PropernessConditionWeight << std::endl;
  os << indent << "RigidityCoefficientImage: " << this->m_RigidityCoefficientImage << std::endl;
  os << indent << "BSplineTransform: " << this->m_BSplineTransform << std::endl;
  os << indent << "RigidityPenaltyTermValue: " << this->m_RigidityPenaltyTermValue << std::endl;
  os << indent << "LinearityConditionValue: " << this->m_LinearityConditionValue << std::endl;
  os << indent << "OrthonormalityConditionValue: " << this->m_OrthonormalityConditionValue << std::endl;
  os << indent << "PropernessConditionValue: " << this->m_PropernessConditionValue << std::endl;
  os << indent << "LinearityConditionGradientMagnitude: " << this->m_LinearityConditionGradientMagnitude << std::endl;
  os << indent << "OrthonormalityConditionGradientMagnitude: " << this->m_OrthonormalityConditionGradientMagnitude
     << std::endl;
  os << indent << "PropernessConditionGradientMagnitude: " << this->m_PropernessConditionGradientMagnitude << std::endl;
  os << indent << "UseLinearityCondition: " << this->m_UseLinearityCondition << std::endl;
  os << indent << "UseOrthonormalityCondition: " << this->m_UseOrthonormalityCondition << std::endl;
  os << indent << "UsePropernessCondition: " << this->m_UsePropernessCondition << std::endl;
  os << indent << "CalculateLinearityCondition: " << this->m_CalculateLinearityCondition << std::endl;
  os << indent << "CalculateOrthonormalityCondition: " << this->m_CalculateOrthonormalityCondition << std::endl;
  os << indent << "CalculatePropernessCondition: " << this->m_CalculatePropernessCondition << std::endl;

} // end PrintSelf()


/**
 * ************************ Create1DOperator *********************
 */

template <class TFixedImage, class TScalarType>
void
TransformRigidityPenaltyTerm<TFixedImage, TScalarType>::Create1DOperator(
  NeighborhoodType &                  F,
  const std::string &                 WhichF,
  const unsigned int                  WhichDimension,
  const CoefficientImageSpacingType & spacing) const
{
  /** Create an operator size and set it in the operator. */
  NeighborhoodSizeType r;
  r.Fill(0U);
  r[WhichDimension - 1] = 1;
  F.SetRadius(r);

  /** Get the image spacing factors that we are going to use. */
  std::vector<double> s(ImageDimension);
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    s[i] = spacing[i];
  }

  /** Create the required operator (neighborhood), depending on
   * WhichF. The operator is either 3x1 or 1x3 in 2D and
   * either 3x1x1 or 1x3x1 or 1x1x3 in 3D.
   */
  if (WhichF == "FA_xi" && WhichDimension == 1)
  {
    /** This case refers to the vector
     * [ B2(3/2)-B2(1/2), B2(1/2)-B2(-1/2), B2(-1/2)-B2(-3/2) ],
     * which is something like 1/2 * [-1 0 1].
     */
    F[0] = -0.5 / s[0];
    F[1] = 0.0;
    F[2] = 0.5 / s[0];
  }
  else if (WhichF == "FA_xi" && WhichDimension == 2)
  {
    /** This case refers to the vector
     * [ B3(-1), B3(0), B3(1) ],
     * which is something like 1/6 * [1 4 1].
     */
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FA_xi" && WhichDimension == 3)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FB_xi" && WhichDimension == 1)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FB_xi" && WhichDimension == 2)
  {
    F[0] = -0.5 / s[1];
    F[1] = 0.0;
    F[2] = 0.5 / s[1];
  }
  else if (WhichF == "FB_xi" && WhichDimension == 3)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FC_xi" && WhichDimension == 1)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FC_xi" && WhichDimension == 2)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FC_xi" && WhichDimension == 3)
  {
    F[0] = -0.5 / s[2];
    F[1] = 0.0;
    F[2] = 0.5 / s[2];
  }
  else if (WhichF == "FD_xi" && WhichDimension == 1)
  {
    /** This case refers to the vector
     * [ B1(0), -2*B1(0), B1(0)],
     * which is something like 1/2 * [1 -2 1].
     */
    F[0] = 0.5 / (s[0] * s[0]);
    F[1] = -1.0 / (s[0] * s[0]);
    F[2] = 0.5 / (s[0] * s[0]);
  }
  else if (WhichF == "FD_xi" && WhichDimension == 2)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FD_xi" && WhichDimension == 3)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FE_xi" && WhichDimension == 1)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FE_xi" && WhichDimension == 2)
  {
    F[0] = 0.5 / (s[1] * s[1]);
    F[1] = -1.0 / (s[1] * s[1]);
    F[2] = 0.5 / (s[1] * s[1]);
  }
  else if (WhichF == "FE_xi" && WhichDimension == 3)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FF_xi" && WhichDimension == 1)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FF_xi" && WhichDimension == 2)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FF_xi" && WhichDimension == 3)
  {
    F[0] = 0.5 / (s[2] * s[2]);
    F[1] = -1.0 / (s[2] * s[2]);
    F[2] = 0.5 / (s[2] * s[2]);
  }
  else if (WhichF == "FG_xi" && WhichDimension == 1)
  {
    F[0] = -0.5 / (s[0] * s[1]);
    F[1] = 0.0;
    F[2] = 0.5 / (s[0] * s[1]);
  }
  else if (WhichF == "FG_xi" && WhichDimension == 2)
  {
    F[0] = -0.5 / (s[0] * s[1]);
    F[1] = 0.0;
    F[2] = 0.5 / (s[0] * s[1]);
  }
  else if (WhichF == "FG_xi" && WhichDimension == 3)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FH_xi" && WhichDimension == 1)
  {
    F[0] = -0.5 / (s[0] * s[2]);
    F[1] = 0.0;
    F[2] = 0.5 / (s[0] * s[2]);
  }
  else if (WhichF == "FH_xi" && WhichDimension == 2)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FH_xi" && WhichDimension == 3)
  {
    F[0] = -0.5 / (s[0] * s[2]);
    F[1] = 0.0;
    F[2] = 0.5 / (s[0] * s[2]);
  }
  else if (WhichF == "FI_xi" && WhichDimension == 1)
  {
    F[0] = 1.0 / 6.0;
    F[1] = 4.0 / 6.0;
    F[2] = 1.0 / 6.0;
  }
  else if (WhichF == "FI_xi" && WhichDimension == 2)
  {
    F[0] = -0.5 / (s[1] * s[2]);
    F[1] = 0.0;
    F[2] = 0.5 / (s[1] * s[2]);
  }
  else if (WhichF == "FI_xi" && WhichDimension == 3)
  {
    F[0] = -0.5 / (s[1] * s[2]);
    F[1] = 0.0;
    F[2] = 0.5 / (s[1] * s[2]);
  }
  else
  {
    /** Throw an exception. */
    itkExceptionMacro(<< "Can not create this type of operator.");
  }

} // end Create1DOperator()


/**
 * ************************** FilterSeparable ********************
 */

template <class TFixedImage, class TScalarType>
auto
TransformRigidityPenaltyTerm<TFixedImage, TScalarType>::FilterSeparable(
  const CoefficientImageType *          image,
  const std::vector<NeighborhoodType> & Operators) const -> CoefficientImagePointer
{
  /** Create filters, supply them with boundary conditions and operators. */
  std::vector<typename NOIFType::Pointer> filters(ImageDimension);
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    filters[i] = NOIFType::New();
    filters[i]->SetOperator(Operators[i]);
  }

  /** Set up the mini-pipline. */
  filters[0]->SetInput(image);
  for (unsigned int i = 1; i < ImageDimension; ++i)
  {
    filters[i]->SetInput(filters[i - 1]->GetOutput());
  }

  /** Execute the mini-pipeline. */
  filters[ImageDimension - 1]->Update();

  /** Return the filtered image. */
  return filters[ImageDimension - 1]->GetOutput();

} // end FilterSeparable()


/**
 * ************************ CreateNDOperator *********************
 */

template <class TFixedImage, class TScalarType>
void
TransformRigidityPenaltyTerm<TFixedImage, TScalarType>::CreateNDOperator(
  NeighborhoodType &                  F,
  const std::string &                 WhichF,
  const CoefficientImageSpacingType & spacing) const
{
  /** Create an operator size and set it in the operator. */
  NeighborhoodSizeType r;
  r.Fill(1);
  F.SetRadius(r);

  /** Get the image spacing factors that we are going to use. */
  std::vector<double> s(ImageDimension);
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    s[i] = spacing[i];
  }

  /** Create the required operator (neighborhood), depending on
   * WhichF. The operator is either 3x3 in 2D or 3x3x3 in 3D.
   */
  if (WhichF == "FA")
  {
    if (ImageDimension == 2)
    {
      F[0] = 1.0 / 12.0 / s[0];
      F[1] = 0.0;
      F[2] = -1.0 / 12.0 / s[0];
      F[3] = 1.0 / 3.0 / s[0];
      F[4] = 0.0;
      F[5] = -1.0 / 3.0 / s[0];
      F[6] = 1.0 / 12.0 / s[0];
      F[7] = 0.0;
      F[8] = -1.0 / 12.0 / s[0];
    }
    else if (ImageDimension == 3)
    {
      /** Fill the operator. First slice. */
      F[0] = 1.0 / 72.0 / s[0];
      F[1] = 0.0;
      F[2] = -1.0 / 72.0 / s[0];
      F[3] = 1.0 / 18.0 / s[0];
      F[4] = 0.0;
      F[5] = -1.0 / 18.0 / s[0];
      F[6] = 1.0 / 72.0 / s[0];
      F[7] = 0.0;
      F[8] = -1.0 / 72.0 / s[0];
      /** Second slice. */
      F[9] = 1.0 / 18.0 / s[0];
      F[10] = 0.0;
      F[11] = -1.0 / 18.0 / s[0];
      F[12] = 2.0 / 9.0 / s[0];
      F[13] = 0.0;
      F[14] = -2.0 / 9.0 / s[0];
      F[15] = 1.0 / 18.0 / s[0];
      F[16] = 0.0;
      F[17] = -1.0 / 18.0 / s[0];
      /** Third slice. */
      F[18] = 1.0 / 72.0 / s[0];
      F[19] = 0.0;
      F[20] = -1.0 / 72.0 / s[0];
      F[21] = 1.0 / 18.0 / s[0];
      F[22] = 0.0;
      F[23] = -1.0 / 18.0 / s[0];
      F[24] = 1.0 / 72.0 / s[0];
      F[25] = 0.0;
      F[26] = -1.0 / 72.0 / s[0];
    }
  }
  else if (WhichF == "FB")
  {
    if (ImageDimension == 2)
    {
      F[0] = 1.0 / 12.0 / s[1];
      F[1] = 1.0 / 3.0 / s[1];
      F[2] = 1.0 / 12.0 / s[1];
      F[3] = 0.0;
      F[4] = 0.0;
      F[5] = 0.0;
      F[6] = -1.0 / 12.0 / s[1];
      F[7] = -1.0 / 3.0 / s[1];
      F[8] = -1.0 / 12.0 / s[1];
    }
    else if (ImageDimension == 3)
    {
      /** Fill the operator. First slice. */
      F[0] = 1.0 / 72.0 / s[1];
      F[1] = 1.0 / 18.0 / s[1];
      F[2] = 1.0 / 72.0 / s[1];
      F[3] = 0.0;
      F[4] = 0.0;
      F[5] = 0.0;
      F[6] = -1.0 / 72.0 / s[1];
      F[7] = -1.0 / 18.0 / s[1];
      F[8] = -1.0 / 72.0 / s[1];
      /** Second slice. */
      F[9] = 1.0 / 18.0 / s[1];
      F[10] = 2.0 / 9.0 / s[1];
      F[11] = 1.0 / 18.0 / s[1];
      F[12] = 0.0;
      F[13] = 0.0;
      F[14] = 0.0;
      F[15] = -1.0 / 18.0 / s[1];
      F[16] = -2.0 / 9.0 / s[1];
      F[17] = -1.0 / 18.0 / s[1];
      /** Third slice. */
      F[18] = 1.0 / 72.0 / s[1];
      F[19] = 1.0 / 18.0 / s[1];
      F[20] = 1.0 / 72.0 / s[1];
      F[21] = 0.0;
      F[22] = 0.0;
      F[23] = 0.0;
      F[24] = -1.0 / 72.0 / s[1];
      F[25] = -1.0 / 18.0 / s[1];
      F[26] = -1.0 / 72.0 / s[1];
    }
  }
  else if (WhichF == "FC")
  {
    if (ImageDimension == 2)
    {
      /** Not appropriate. Throw an exception. */
      itkExceptionMacro(<< "This type of operator (FC) is not appropriate in 2D.");
    }
    else if (ImageDimension == 3)
    {
      /** Fill the operator. First slice. */
      F[0] = 1.0 / 72.0 / s[2];
      F[1] = 1.0 / 18.0 / s[2];
      F[2] = 1.0 / 72.0 / s[2];
      F[3] = 1.0 / 18.0 / s[2];
      F[4] = 2.0 / 9.0 / s[2];
      F[5] = 1.0 / 18.0 / s[2];
      F[6] = 1.0 / 72.0 / s[2];
      F[7] = 1.0 / 18.0 / s[2];
      F[8] = 1.0 / 72.0 / s[2];
      /** Second slice. */
      F[9] = 0.0;
      F[10] = 0.0;
      F[11] = 0.0;
      F[12] = 0.0;
      F[13] = 0.0;
      F[14] = 0.0;
      F[15] = 0.0;
      F[16] = 0.0;
      F[17] = 0.0;
      /** Third slice. */
      F[18] = -1.0 / 72.0 / s[2];
      F[19] = -1.0 / 18.0 / s[2];
      F[20] = -1.0 / 72.0 / s[2];
      F[21] = -1.0 / 18.0 / s[2];
      F[22] = -2.0 / 9.0 / s[2];
      F[23] = -1.0 / 18.0 / s[2];
      F[24] = -1.0 / 72.0 / s[2];
      F[25] = -1.0 / 18.0 / s[2];
      F[26] = -1.0 / 72.0 / s[2];
    }
  }
  else if (WhichF == "FD")
  {
    if (ImageDimension == 2)
    {
      double sp = s[0] * s[0];
      F[0] = 1.0 / 12.0 / sp;
      F[1] = -1.0 / 6.0 / sp;
      F[2] = 1.0 / 12.0 / sp;
      F[3] = 1.0 / 3.0 / sp;
      F[4] = -2.0 / 3.0 / sp;
      F[5] = 1.0 / 3.0 / sp;
      F[6] = 1.0 / 12.0 / sp;
      F[7] = -1.0 / 6.0 / sp;
      F[8] = 1.0 / 12.0 / sp;
    }
    else if (ImageDimension == 3)
    {
      double sp = s[0] * s[0];
      /** Fill the operator. First slice. */
      F[0] = 1.0 / 72.0 / sp;
      F[1] = -1.0 / 36.0 / sp;
      F[2] = 1.0 / 72.0 / sp;
      F[3] = 1.0 / 18.0 / sp;
      F[4] = -1.0 / 9.0 / sp;
      F[5] = 1.0 / 18.0 / sp;
      F[6] = 1.0 / 72.0 / sp;
      F[7] = -1.0 / 36.0 / sp;
      F[8] = 1.0 / 72.0 / sp;
      /** Second slice. */
      F[9] = 1.0 / 18.0 / sp;
      F[10] = -1.0 / 9.0 / sp;
      F[11] = 1.0 / 18.0 / sp;
      F[12] = 2.0 / 9.0 / sp;
      F[13] = -4.0 / 9.0 / sp;
      F[14] = 2.0 / 9.0 / sp;
      F[15] = 1.0 / 18.0 / sp;
      F[16] = -1.0 / 9.0 / sp;
      F[17] = 1.0 / 18.0 / sp;
      /** Third slice. */
      F[18] = 1.0 / 72.0 / sp;
      F[19] = -1.0 / 36.0 / sp;
      F[20] = 1.0 / 72.0 / sp;
      F[21] = 1.0 / 18.0 / sp;
      F[22] = -1.0 / 9.0 / sp;
      F[23] = 1.0 / 18.0 / sp;
      F[24] = 1.0 / 72.0 / sp;
      F[25] = -1.0 / 36.0 / sp;
      F[26] = 1.0 / 72.0 / sp;
    }
  }
  else if (WhichF == "FE")
  {
    if (ImageDimension == 2)
    {
      double sp = s[1] * s[1];
      F[0] = 1.0 / 12.0 / sp;
      F[1] = 1.0 / 3.0 / sp;
      F[2] = 1.0 / 12.0 / sp;
      F[3] = -1.0 / 6.0 / sp;
      F[4] = -2.0 / 3.0 / sp;
      F[5] = -1.0 / 6.0 / sp;
      F[6] = 1.0 / 12.0 / sp;
      F[7] = 1.0 / 3.0 / sp;
      F[8] = 1.0 / 12.0 / sp;
    }
    else if (ImageDimension == 3)
    {
      double sp = s[1] * s[1];
      /** Fill the operator. First slice. */
      F[0] = 1.0 / 72.0 / sp;
      F[1] = 1.0 / 18.0 / sp;
      F[2] = 1.0 / 72.0 / sp;
      F[3] = -1.0 / 36.0 / sp;
      F[4] = -1.0 / 9.0 / sp;
      F[5] = -1.0 / 36.0 / sp;
      F[6] = 1.0 / 72.0 / sp;
      F[7] = 1.0 / 18.0 / sp;
      F[8] = 1.0 / 72.0 / sp;
      /** Second slice. */
      F[9] = 1.0 / 18.0 / sp;
      F[10] = 2.0 / 9.0 / sp;
      F[11] = 1.0 / 18.0 / sp;
      F[12] = -1.0 / 9.0 / sp;
      F[13] = -4.0 / 9.0 / sp;
      F[14] = -1.0 / 9.0 / sp;
      F[15] = 1.0 / 18.0 / sp;
      F[16] = 2.0 / 9.0 / sp;
      F[17] = 1.0 / 18.0 / sp;
      /** Third slice. */
      F[18] = 1.0 / 72.0 / sp;
      F[19] = 1.0 / 18.0 / sp;
      F[20] = 1.0 / 72.0 / sp;
      F[21] = -1.0 / 36.0 / sp;
      F[22] = -1.0 / 9.0 / sp;
      F[23] = -1.0 / 36.0 / sp;
      F[24] = 1.0 / 72.0 / sp;
      F[25] = 1.0 / 18.0 / sp;
      F[26] = 1.0 / 72.0 / sp;
    }
  }
  else if (WhichF == "FF")
  {
    if (ImageDimension == 2)
    {
      /** Not appropriate. Throw an exception. */
      itkExceptionMacro(<< "This type of operator (FF) is not appropriate in 2D.");
    }
    else if (ImageDimension == 3)
    {
      double sp = s[2] * s[2];
      /** Fill the operator. First slice. */
      F[0] = 1.0 / 72.0 / sp;
      F[1] = 1.0 / 18.0 / sp;
      F[2] = 1.0 / 72.0 / sp;
      F[3] = 1.0 / 18.0 / sp;
      F[4] = 2.0 / 9.0 / sp;
      F[5] = 1.0 / 18.0 / sp;
      F[6] = 1.0 / 72.0 / sp;
      F[7] = 1.0 / 18.0 / sp;
      F[8] = 1.0 / 72.0 / sp;
      /** Second slice. */
      F[9] = -1.0 / 39.0 / sp;
      F[10] = -1.0 / 9.0 / sp;
      F[11] = -1.0 / 36.0 / sp;
      F[12] = -1.0 / 9.0 / sp;
      F[13] = -4.0 / 9.0 / sp;
      F[14] = -1.0 / 9.0 / sp;
      F[15] = -1.0 / 36.0 / sp;
      F[16] = -1.0 / 9.0 / sp;
      F[17] = -1.0 / 36.0 / sp;
      /** Third slice. */
      F[18] = 1.0 / 72.0 / sp;
      F[19] = 1.0 / 18.0 / sp;
      F[20] = 1.0 / 72.0 / sp;
      F[21] = 1.0 / 18.0 / sp;
      F[22] = 2.0 / 9.0 / sp;
      F[23] = 1.0 / 18.0 / sp;
      F[24] = 1.0 / 72.0 / sp;
      F[25] = 1.0 / 18.0 / sp;
      F[26] = 1.0 / 72.0 / sp;
    }
  }
  else if (WhichF == "FG")
  {
    if (ImageDimension == 2)
    {
      double sp = s[0] * s[1];
      F[0] = 1.0 / 4.0 / sp;
      F[1] = 0.0;
      F[2] = -1.0 / 4.0 / sp;
      F[3] = 0.0;
      F[4] = 0.0;
      F[5] = 0.0;
      F[6] = -1.0 / 4.0 / sp;
      F[7] = 0.0;
      F[8] = 1.0 / 4.0 / sp;
    }
    else if (ImageDimension == 3)
    {
      double sp = s[0] * s[1];
      /** Fill the operator. First slice. */
      F[0] = 1.0 / 24.0 / sp;
      F[1] = 0.0;
      F[2] = -1.0 / 24.0 / sp;
      F[3] = 0.0;
      F[4] = 0.0;
      F[5] = 0.0;
      F[6] = -1.0 / 24.0 / sp;
      F[7] = 0.0;
      F[8] = 1.0 / 24.0 / sp;
      /** Second slice. */
      F[9] = 1.0 / 6.0 / sp;
      F[10] = 0.0;
      F[11] = -1.0 / 6.0 / sp;
      F[12] = 0.0;
      F[13] = 0.0;
      F[14] = 0.0;
      F[15] = -1.0 / 6.0 / sp;
      F[16] = 0.0;
      F[17] = 1.0 / 6.0 / sp;
      /** Third slice. */
      F[18] = 1.0 / 24.0 / sp;
      F[19] = 0.0;
      F[20] = -1.0 / 24.0 / sp;
      F[21] = 0.0;
      F[22] = 0.0;
      F[23] = 0.0;
      F[24] = -1.0 / 24.0 / sp;
      F[25] = 0.0;
      F[26] = 1.0 / 24.0 / sp;
    }
  }
  else if (WhichF == "FH")
  {
    if (ImageDimension == 2)
    {
      /** Not appropriate. Throw an exception. */
      itkExceptionMacro(<< "This type of operator (FH) is not appropriate in 2D.");
    }
    else if (ImageDimension == 3)
    {
      double sp = s[0] * s[2];
      /** Fill the operator. First slice. */
      F[0] = 1.0 / 24.0 / sp;
      F[1] = 0.0;
      F[2] = -1.0 / 24.0 / sp;
      F[3] = 1.0 / 6.0 / sp;
      F[4] = 0.0;
      F[5] = -1.0 / 6.0 / sp;
      F[6] = 1.0 / 24.0 / sp;
      F[7] = 0.0;
      F[8] = -1.0 / 24.0 / sp;
      /** Second slice. */
      F[9] = 0.0;
      F[10] = 0.0;
      F[11] = 0.0;
      F[12] = 0.0;
      F[13] = 0.0;
      F[14] = 0.0;
      F[15] = 0.0;
      F[16] = 0.0;
      F[17] = 0.0;
      /** Third slice. */
      F[18] = -1.0 / 24.0 / sp;
      F[19] = 0.0;
      F[20] = 1.0 / 24.0 / sp;
      F[21] = -1.0 / 6.0 / sp;
      F[22] = 0.0;
      F[23] = 1.0 / 6.0 / sp;
      F[24] = -1.0 / 24.0 / sp;
      F[25] = 0.0;
      F[26] = 1.0 / 24.0 / sp;
    }
  }
  else if (WhichF == "FI")
  {
    if (ImageDimension == 2)
    {
      /** Not appropriate. Throw an exception. */
      itkExceptionMacro(<< "This type of operator (FI) is not appropriate in 2D.");
    }
    else if (ImageDimension == 3)
    {
      double sp = s[1] * s[2];
      /** Fill the operator. First slice. */
      F[0] = 1.0 / 24.0 / sp;
      F[1] = 1.0 / 6.0 / sp;
      F[2] = 1.0 / 24.0 / sp;
      F[3] = 0.0;
      F[4] = 0.0;
      F[5] = 0.0;
      F[6] = -1.0 / 24.0 / sp;
      F[7] = -1.0 / 6.0 / sp;
      F[8] = -1.0 / 24.0 / sp;
      /** Second slice. */
      F[9] = 0.0;
      F[10] = 0.0;
      F[11] = 0.0;
      F[12] = 0.0;
      F[13] = 0.0;
      F[14] = 0.0;
      F[15] = 0.0;
      F[16] = 0.0;
      F[17] = 0.0;
      /** Third slice. */
      F[18] = -1.0 / 24.0 / sp;
      F[19] = -1.0 / 6.0 / sp;
      F[20] = -1.0 / 24.0 / sp;
      F[21] = 0.0;
      F[22] = 0.0;
      F[23] = 0.0;
      F[24] = 1.0 / 24.0 / sp;
      F[25] = 1.0 / 6.0 / sp;
      F[26] = 1.0 / 24.0 / sp;
    }
  }
  else
  {
    /** Throw an exception. */
    itkExceptionMacro(<< "Can not create this type of operator.");
  }

} // end CreateNDOperator()


} // end namespace itk

#endif // #ifndef itkTransformRigidityPenaltyTerm_hxx
