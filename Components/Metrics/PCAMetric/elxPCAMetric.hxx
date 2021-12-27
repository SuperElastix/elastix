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
#ifndef elxPCAMetric_hxx
#define elxPCAMetric_hxx

#include "elxPCAMetric.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
PCAMetric<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of PCAMetric metric took: " << static_cast<long>(timer.GetMean() * 1000) << " ms."
         << std::endl;

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
PCAMetric<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level. */
  unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();

  unsigned int NumEigenValues = 6;
  this->GetConfiguration()->ReadParameter(NumEigenValues, "NumEigenValues", this->GetComponentLabel(), level, 0);
  this->SetNumEigenValues(NumEigenValues);

  /** Get and set if we want to subtract the mean from the derivative. */
  bool subtractMean = false;
  this->GetConfiguration()->ReadParameter(subtractMean, "SubtractMean", this->GetComponentLabel(), 0, 0);
  this->SetSubtractMean(subtractMean);

  /** Get and set the number of additional samples sampled at the fixed timepoint.  */
  //    unsigned int numAdditionalSamplesFixed = 0;
  //    this->GetConfiguration()->ReadParameter( numAdditionalSamplesFixed,
  //      "NumAdditionalSamplesFixed", this->GetComponentLabel(), level, 0 );
  //    this->SetNumAdditionalSamplesFixed( numAdditionalSamplesFixed );

  /** Get and set the fixed timepoint number. */
  //    unsigned int reducedDimensionIndex = 0;
  //    this->GetConfiguration()->ReadParameter(
  //        reducedDimensionIndex, "ReducedDimensionIndex",
  //        this->GetComponentLabel(), 0, 0 );
  //    this->SetReducedDimensionIndex( reducedDimensionIndex );

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
    elxout << "Multiplying moving image derivatives by: " << movingImageDerivativeScales << std::endl;
  }

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
          const ReducedDimensionBSplineTransformBaseType * testPtr4 =
            dynamic_cast<const ReducedDimensionBSplineTransformBaseType *>(testPtr3->GetSubTransform(0).GetPointer());
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
  elxout << "end BeforeEachResolution" << std::endl;

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxPCAMetric_hxx
