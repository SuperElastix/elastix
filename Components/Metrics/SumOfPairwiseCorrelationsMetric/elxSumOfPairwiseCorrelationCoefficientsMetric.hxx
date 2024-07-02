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
#include <itkDeref.h>

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

  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  if (configuration.HasParameter("SubtractMean"))
  {
    log::warn(std::string("WARNING: From elastix version 5.2, the ") + elxGetClassNameStatic() +
              " parameter `SubtractMean` (default \"false\") is "
              "replaced with `UseZeroAverageDisplacementConstraint` "
              "(default \"true\").");
  }

  timer.Stop();
  log::info(std::ostringstream{} << "Initialization of SumOfPairwiseCorrelationCoefficientsMetric metric took: "
                                 << static_cast<long>(timer.GetMean() * 1000) << " ms.");

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
SumOfPairwiseCorrelationCoefficientsMetric<TElastix>::BeforeRegistration()
{
  /** Check if this elastix object has a transform. (If so, it must be a combination transform.) */
  if (CombinationTransformType * const combinationTransform{
        BaseComponent::AsITKBaseType(this->GetElastix()->GetElxTransformBase()) })
  {
    auto * const currentTransform = combinationTransform->GetModifiableCurrentTransform();

    /** Check for B-spline transform. */
    if (const auto bsplineTransform = dynamic_cast<BSplineTransformBaseType *>(currentTransform))
    {
      this->SetGridSize(bsplineTransform->GetGridRegion().GetSize());
    }
    else
    {
      /** Check for stack transform. */
      if (const auto stackTransform = dynamic_cast<StackTransformType *>(currentTransform))
      {
        /** Set itk member variable. */
        this->SetTransformIsStackTransform(true);

        // Return early, now that the current transform is a stack transform.
        return;
      }
    }
  }
  // If the current transform would have been a stack transform, the function would have returned earlier.
  this->SetTransformIsStackTransform(false);
}


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
SumOfPairwiseCorrelationCoefficientsMetric<TElastix>::BeforeEachResolution()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());
  const std::string     componentLabel = BaseComponent::GetComponentLabel();

  /** Get the current resolution level. */
  unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();

  /** Get and set if we want to subtract the mean from the derivative. */
  bool useZeroAverageDisplacementConstraint = true;
  // The parameter name "SubtractMean" is obsolete, so just use it as initial value, for backward compatibility.
  configuration.ReadParameter(useZeroAverageDisplacementConstraint, "SubtractMean", componentLabel, 0, 0);
  configuration.ReadParameter(
    useZeroAverageDisplacementConstraint, "UseZeroAverageDisplacementConstraint", componentLabel, 0, 0);
  this->SetUseZeroAverageDisplacementConstraint(useZeroAverageDisplacementConstraint);

  /** Get and set the number of additional samples sampled at the fixed timepoint.  */
  unsigned int numAdditionalSamplesFixed = 0;
  configuration.ReadParameter(numAdditionalSamplesFixed, "NumAdditionalSamplesFixed", componentLabel, level, 0);
  this->SetNumAdditionalSamplesFixed(numAdditionalSamplesFixed);

  /** Get and set the fixed timepoint number. */
  unsigned int reducedDimensionIndex = 0;
  configuration.ReadParameter(reducedDimensionIndex, "ReducedDimensionIndex", componentLabel, 0, 0);
  this->SetReducedDimensionIndex(reducedDimensionIndex);

  /** Set moving image derivative scales. */
  this->SetUseMovingImageDerivativeScales(false);
  MovingImageDerivativeScalesType movingImageDerivativeScales;
  bool                            usescales = true;
  for (unsigned int i = 0; i < MovingImageDimension; ++i)
  {
    usescales =
      usescales && configuration.ReadParameter(
                     movingImageDerivativeScales[i], "MovingImageDerivativeScales", componentLabel, i, -1, true);
  }
  if (usescales)
  {
    this->SetUseMovingImageDerivativeScales(true);
    this->SetMovingImageDerivativeScales(movingImageDerivativeScales);
    log::info(std::ostringstream{} << "Multiplying moving image derivatives by: " << movingImageDerivativeScales);
  }

  log::info("end BeforeEachResolution");

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxSumOfPairwiseCorrelationCoefficientsMetric_hxx
