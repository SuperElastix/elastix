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
#ifndef elxVarianceOverLastDimensionMetric_hxx
#define elxVarianceOverLastDimensionMetric_hxx

#include "elxVarianceOverLastDimensionMetric.h"
#include "itkTimeProbe.h"
#include <itkDeref.h>

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <typename TElastix>
void
VarianceOverLastDimensionMetric<TElastix>::Initialize()
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
  log::info(std::ostringstream{} << "Initialization of VarianceOverLastDimensionMetric metric took: "
                                 << static_cast<long>(timer.GetMean() * 1000) << " ms.");

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <typename TElastix>
void
VarianceOverLastDimensionMetric<TElastix>::BeforeRegistration()
{
  /** Check that the direction cosines are structured like
   *       [ dc  dc  0 ]
   *  dc = [ dc  dc  0 ]
   *       [  0   0  1 ]
   */
  using DirectionType = typename FixedImageType::DirectionType;
  DirectionType dc = this->GetElastix()->GetFixedImage()->GetDirection();

  bool dcValid = true;
  for (unsigned int i = 0; i < FixedImageDimension - 1; ++i)
  {
    dcValid &= (dc[FixedImageDimension - 1][i] == 0);
    dcValid &= (dc[i][FixedImageDimension - 1] == 0);
  }
  dcValid &= (dc[FixedImageDimension - 1][FixedImageDimension - 1] == 1);

  if (!dcValid)
  {
    itkExceptionMacro("\nERROR: the direction cosines matrix of the fixed image is invalid!\n\n"
                      << "  The VarianceOverLastDimensionMetric expects the last dimension to represent\n"
                      << "  time and therefore requires a direction cosines matrix of the form:\n"
                      << "       [ . . 0 ]\n"
                      << "  dc = [ . . 0 ]\n"
                      << "       [ 0 0 1 ]");
  }

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

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <typename TElastix>
void
VarianceOverLastDimensionMetric<TElastix>::BeforeEachResolution()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());
  const std::string     componentLabel = BaseComponent::GetComponentLabel();

  /** Get the current resolution level. */
  unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();

  /** Get and set the random sampling in the last dimension. */
  bool useRandomSampling = false;
  configuration.ReadParameter(useRandomSampling, "SampleLastDimensionRandomly", componentLabel, level, 0);
  this->SetSampleLastDimensionRandomly(useRandomSampling);

  /** Get and set if we want to subtract the mean from the derivative. */
  bool useZeroAverageDisplacementConstraint = true;
  // The parameter name "SubtractMean" is obsolete, so just use it as initial value, for backward compatibility.
  configuration.ReadParameter(useZeroAverageDisplacementConstraint, "SubtractMean", componentLabel, 0, 0);
  configuration.ReadParameter(
    useZeroAverageDisplacementConstraint, "UseZeroAverageDisplacementConstraint", componentLabel, 0, 0);
  this->SetUseZeroAverageDisplacementConstraint(useZeroAverageDisplacementConstraint);

  /** Get and set the number of random samples for the last dimension. */
  int numSamplesLastDimension = 10;
  configuration.ReadParameter(numSamplesLastDimension, "NumSamplesLastDimension", componentLabel, level, 0);
  this->SetNumSamplesLastDimension(numSamplesLastDimension);

  /** Get and set the number of additional samples sampled at the fixed time point.  */
  unsigned int numAdditionalSamplesFixed = 0;
  configuration.ReadParameter(numAdditionalSamplesFixed, "NumAdditionalSamplesFixed", componentLabel, level, 0);
  this->SetNumAdditionalSamplesFixed(numAdditionalSamplesFixed);

  /** Get and set the fixed timepoint number. */
  unsigned int reducedDimensionIndex = 0;
  configuration.ReadParameter(reducedDimensionIndex, "ReducedDimensionIndex", componentLabel, 0, 0);
  this->SetReducedDimensionIndex(reducedDimensionIndex);

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxVarianceOverLastDimensionMetric_hxx
