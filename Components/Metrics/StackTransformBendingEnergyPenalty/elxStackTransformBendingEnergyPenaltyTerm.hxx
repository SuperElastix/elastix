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
#ifndef __elxStackTransformBendingEnergyPenaltyTerm_HXX__
#define __elxStackTransformBendingEnergyPenaltyTerm_HXX__

#include "elxStackTransformBendingEnergyPenaltyTerm.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
StackTransformBendingEnergyPenalty<TElastix>::Initialize(void)
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  log::info(std::ostringstream{} << "Initialization of StackTransformBendingEnergy metric took: "
                                 << static_cast<long>(timer.GetMean() * 1000) << " ms.");

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
StackTransformBendingEnergyPenalty<TElastix>::BeforeRegistration(void)
{
  bool subtractMean = false;
  this->GetConfiguration()->ReadParameter(subtractMean, "SubtractMean", this->GetComponentLabel(), 0, 0);
  this->SetSubtractMean(subtractMean);
} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
StackTransformBendingEnergyPenalty<TElastix>::BeforeEachResolution(void)
{
  /** Get the current resolution level. */
  unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();

  /** Check if this transform is a B-spline transform. */
  CombinationTransformType * testPtr1 =
    dynamic_cast<CombinationTransformType *>(this->GetElastix()->GetElxTransformBase());
  if (testPtr1)
  {
    /** Check for B-spline transform. */
    BSplineTransformBaseType * testPtr2 = dynamic_cast<BSplineTransformBaseType *>(testPtr1->GetCurrentTransform());
    if (testPtr2)
    {
      this->SetGridSize(testPtr2->GetGridRegion().GetSize());
      this->SetTransformIsBSpline(true);
      itkExceptionMacro(<< "This metric can only be used in combination with a StackTransform");
    }
    else
    {
      StackTransformType * testPtr3 = dynamic_cast<StackTransformType *>(testPtr1->GetCurrentTransform());
      if (testPtr3)
      {
        this->SetTransformIsStackTransform(true);

        if (testPtr3->GetNumberOfSubTransforms() > 0)
        {
          ReducedDimensionBSplineTransformBaseType * testPtr4 =
            dynamic_cast<ReducedDimensionBSplineTransformBaseType *>(testPtr3->GetSubTransform(0).GetPointer());
          if (testPtr4)
          {
            FixedImageSizeType gridSize;
            gridSize.Fill(testPtr3->GetNumberOfSubTransforms());
            this->SetGridSize(gridSize);
            this->SetSubTransformIsBSpline(true);
          }
        }
      }
      else
      {
        itkExceptionMacro(<< "This metric can only be used in combination with a StackTransform");
      }
    }
  }

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef __elxStackTransformBendingEnergyPenaltyTerm_HXX__
