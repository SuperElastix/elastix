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

#include <cassert>

namespace itk
{
template <unsigned int NDimension>
class ITK_TEMPLATE_EXPORT AbstractBSplineStackTransform
  : public itk::StackTransform<elx::ElastixBase::CoordRepType, NDimension, NDimension>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AbstractBSplineStackTransform);

  using Self = AbstractBSplineStackTransform;
  using Superclass = itk::StackTransform<elx::ElastixBase::CoordRepType, NDimension, NDimension>;
  using Pointer = itk::SmartPointer<AbstractBSplineStackTransform>;

  itkTypeMacro(AbstractBSplineStackTransform, Superclass);

  enum
  {
    // The minimum, maximum, and default supported spline order.
    minSplineOrder = 1,
    maxSplineOrder = 3,
    defaultSplineOrder = maxSplineOrder
  };


  virtual unsigned
  GetSplineOrder() const = 0;

protected:
  /** Default-constructor */
  AbstractBSplineStackTransform() = default;

  /** Destructor */
  ~AbstractBSplineStackTransform() override = default;
};


template <unsigned NDimension, unsigned VSplineOrder>
class ITK_TEMPLATE_EXPORT BSplineStackTransform : public AbstractBSplineStackTransform<NDimension>
{
private:
  using CoordRepType = elx::ElastixBase::CoordRepType;

public:
  ITK_DISALLOW_COPY_AND_MOVE(BSplineStackTransform);

  using Self = BSplineStackTransform;
  using Superclass = AbstractBSplineStackTransform<NDimension>;
  using Pointer = itk::SmartPointer<BSplineStackTransform>;
  itkNewMacro(Self);
  itkTypeMacro(BSplineStackTransform, Superclass);

  std::string
  GetTransformTypeAsString() const override
  {
    return Superclass::GetTransformTypeAsString() +
           ((VSplineOrder == Superclass::defaultSplineOrder) ? "" : ('_' + std::to_string(VSplineOrder)));
  }

protected:
  /** Default-constructor */
  BSplineStackTransform() = default;

  /** Destructor */
  ~BSplineStackTransform() override = default;

private:
  unsigned
  GetSplineOrder() const override
  {
    return VSplineOrder;
  }

  /** Create a subtransform that may be added to this specific stack. */
  typename Superclass::SubTransformPointer
  CreateSubTransform() const override
  {
    return AdvancedBSplineDeformableTransform<CoordRepType, NDimension - 1, VSplineOrder>::New();
  }
};

} // namespace itk

#endif
