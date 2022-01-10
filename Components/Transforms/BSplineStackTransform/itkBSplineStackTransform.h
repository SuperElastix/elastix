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
#ifndef itkBSplineStackTransform_h
#define itkBSplineStackTransform_h

#include "itkStackTransform.h"
#include "itkAdvancedBSplineDeformableTransform.h"
#include "elxElastixBase.h"

namespace itk
{
template <unsigned int NDimension>
class ITK_TEMPLATE_EXPORT BSplineStackTransform
  : public itk::StackTransform<elx::ElastixBase::CoordRepType, NDimension, NDimension>
{
private:
  using CoordRepType = elx::ElastixBase::CoordRepType;

public:
  ITK_DISALLOW_COPY_AND_MOVE(BSplineStackTransform);

  using Self = BSplineStackTransform;
  using Superclass = itk::StackTransform<CoordRepType, NDimension, NDimension>;
  using Pointer = itk::SmartPointer<BSplineStackTransform>;
  itkNewMacro(Self);
  itkTypeMacro(BSplineStackTransform, Superclass);

  void
  SetSplineOrder(const unsigned newValue)
  {
    m_SplineOrder = newValue;
  }

protected:
  /** Default-constructor */
  BSplineStackTransform() = default;

  /** Destructor */
  ~BSplineStackTransform() override = default;

private:
  /** Create a subtransform that may be added to this specific stack. */
  typename Superclass::SubTransformPointer
  CreateSubTransform() const override
  {
    return AdvancedBSplineDeformableTransformBase<CoordRepType, NDimension - 1>::template Create<
      AdvancedBSplineDeformableTransform>(m_SplineOrder);
  }

  unsigned m_SplineOrder{ 3 };
};

} // namespace itk

#endif
