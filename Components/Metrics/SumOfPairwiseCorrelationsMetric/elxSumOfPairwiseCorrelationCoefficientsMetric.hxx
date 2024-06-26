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
#ifndef elxSumOfPairwiseCorrelationCoefficientsMetric_hxx
#define elxSumOfPairwiseCorrelationCoefficientsMetric_hxx

#include "elxSumOfPairwiseCorrelationCoefficientsMetric.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
SumOfPairwiseCorrelationCoefficientsMetric<TElastix>::Initialize()
{

  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  log::info(std::ostringstream{} << "Initialization of SumOfPairwiseCorrelationCoefficientsMetric metric took: "
                                 << static_cast<long>(timer.GetMean() * 1000) << " ms.");

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
SumOfPairwiseCorrelationCoefficientsMetric<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level. */
  unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();

  /** Get and set if we want to subtract the mean from the derivative. */
  bool subtractMean = false;
  this->GetConfiguration()->ReadParameter(subtractMean, "SubtractMean", this->GetComponentLabel(), 0, 0);
  this->SetSubtractMean(subtractMean);

  /** Get and set the number of additional samples sampled at the fixed timepoint.  */
  unsigned int numAdditionalSamplesFixed = 0;
  this->GetConfiguration()->ReadParameter(
    numAdditionalSamplesFixed, "NumAdditionalSamplesFixed", this->GetComponentLabel(), level, 0);
  this->SetNumAdditionalSamplesFixed(numAdditionalSamplesFixed);

  /** Get and set the fixed timepoint number. */
  unsigned int reducedDimensionIndex = 0;
  this->GetConfiguration()->ReadParameter(
    reducedDimensionIndex, "ReducedDimensionIndex", this->GetComponentLabel(), 0, 0);
  this->SetReducedDimensionIndex(reducedDimensionIndex);

  /** Set moving image derivative scales. */
  this->SetUseMovingImageDerivativeScales(false);
  MovingImageDerivativeScalesType movingImageDerivativeScales;
  bool                            usescales = true;
  for (unsigned int i = 0; i < MovingImageDimension; ++i)
  {
    usescales =
      usescales &&
      this->GetConfiguration()->ReadParameter(
        movingImageDerivativeScales[i], "MovingImageDerivativeScales", this->GetComponentLabel(), i, -1, true);
  }
  if (usescales)
  {
    this->SetUseMovingImageDerivativeScales(true);
    this->SetMovingImageDerivativeScales(movingImageDerivativeScales);
    log::info(std::ostringstream{} << "Multiplying moving image derivatives by: " << movingImageDerivativeScales);
  }

  /** Check if this transform is a B-spline transform. */
  if (CombinationTransformType * const combinationTransform{
        BaseComponent::AsITKBaseType(this->GetElastix()->GetElxTransformBase()) })
  {
    /** Check for B-spline transform. */
    if (const auto bsplineTransform =
          dynamic_cast<const BSplineTransformBaseType *>(combinationTransform->GetCurrentTransform()))
    {
      this->SetGridSize(bsplineTransform->GetGridRegion().GetSize());
    }
    else
    {
      /** Check for stack transform. */
      if (const auto stackTransform =
            dynamic_cast<StackTransformType *>(combinationTransform->GetModifiableCurrentTransform()))
      {
        /** Set itk member variable. */
        this->SetTransformIsStackTransform(true);

        if (stackTransform->GetNumberOfSubTransforms() > 0)
        {
          /** Check if subtransform is a B-spline transform. */
          if (dynamic_cast<ReducedDimensionBSplineTransformBaseType *>(stackTransform->GetSubTransform(0).GetPointer()))
          {
            FixedImageSizeType gridSize;
            gridSize.Fill(stackTransform->GetNumberOfSubTransforms());
            this->SetGridSize(gridSize);
          }
        }
      }
    }
  }
  log::info("end BeforeEachResolution");

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxSumOfPairwiseCorrelationCoefficientsMetric_hxx
