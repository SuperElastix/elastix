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

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
VarianceOverLastDimensionMetric<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of VarianceOverLastDimensionMetric metric took: "
         << static_cast<long>(timer.GetMean() * 1000) << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
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
    itkExceptionMacro(<< "\nERROR: the direction cosines matrix of the fixed image is invalid!\n\n"
                      << "  The VarianceOverLastDimensionMetric expects the last dimension to represent\n"
                      << "  time and therefore requires a direction cosines matrix of the form:\n"
                      << "       [ . . 0 ]\n"
                      << "  dc = [ . . 0 ]\n"
                      << "       [ 0 0 1 ]");
  }

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
VarianceOverLastDimensionMetric<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level. */
  unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();

  /** Get and set the random sampling in the last dimension. */
  bool useRandomSampling = false;
  this->GetConfiguration()->ReadParameter(
    useRandomSampling, "SampleLastDimensionRandomly", this->GetComponentLabel(), level, 0);
  this->SetSampleLastDimensionRandomly(useRandomSampling);

  /** Get and set if we want to subtract the mean from the derivative. */
  bool subtractMean = false;
  this->GetConfiguration()->ReadParameter(subtractMean, "SubtractMean", this->GetComponentLabel(), 0, 0);
  this->SetSubtractMean(subtractMean);

  /** Get and set the number of random samples for the last dimension. */
  int numSamplesLastDimension = 10;
  this->GetConfiguration()->ReadParameter(
    numSamplesLastDimension, "NumSamplesLastDimension", this->GetComponentLabel(), level, 0);
  this->SetNumSamplesLastDimension(numSamplesLastDimension);

  /** Get and set the number of additional samples sampled at the fixed time point.  */
  unsigned int numAdditionalSamplesFixed = 0;
  this->GetConfiguration()->ReadParameter(
    numAdditionalSamplesFixed, "NumAdditionalSamplesFixed", this->GetComponentLabel(), level, 0);
  this->SetNumAdditionalSamplesFixed(numAdditionalSamplesFixed);

  /** Get and set the fixed timepoint number. */
  unsigned int reducedDimensionIndex = 0;
  this->GetConfiguration()->ReadParameter(
    reducedDimensionIndex, "ReducedDimensionIndex", this->GetComponentLabel(), 0, 0);
  this->SetReducedDimensionIndex(reducedDimensionIndex);

  /** Check if this transform is a B-spline transform. */
  CombinationTransformType * testPtr1 = BaseComponent::AsITKBaseType(this->GetElastix()->GetElxTransformBase());
  if (testPtr1)
  {
    /** Check for B-spline transform. */
    const BSplineTransformBaseType * testPtr2 =
      dynamic_cast<const BSplineTransformBaseType *>(testPtr1->GetCurrentTransform());
    if (testPtr2)
    {
      this->SetGridSize(testPtr2->GetGridRegion().GetSize());
    }
    else
    {
      /** Check for stack transform. */
      StackTransformType * testPtr3 = dynamic_cast<StackTransformType *>(testPtr1->GetModifiableCurrentTransform());
      if (testPtr3)
      {
        /** Set itk member variable. */
        this->SetTransformIsStackTransform(true);

        if (testPtr3->GetNumberOfSubTransforms() > 0)
        {
          /** Check if subtransform is a B-spline transform. */
          ReducedDimensionBSplineTransformBaseType * testPtr4 =
            dynamic_cast<ReducedDimensionBSplineTransformBaseType *>(testPtr3->GetSubTransform(0).GetPointer());
          if (testPtr4)
          {
            FixedImageSizeType gridSize;
            gridSize.Fill(testPtr3->GetNumberOfSubTransforms());
            this->SetGridSize(gridSize);
          }
        }
      }
    }
  }

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxVarianceOverLastDimensionMetric_hxx
