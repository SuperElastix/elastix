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
#ifndef itkEulerStackTransform_h
#define itkEulerStackTransform_h

#include "itkStackTransform.h"
#include "itkEulerTransform.h"
#include "elxElastixBase.h"

namespace itk
{
template <unsigned int NDimension>
class ITK_TEMPLATE_EXPORT EulerStackTransform
  : public itk::StackTransform<elx::ElastixBase::CoordRepType, NDimension, NDimension>
{
private:
  using CoordRepType = elx::ElastixBase::CoordRepType;

public:
  ITK_DISALLOW_COPY_AND_MOVE(EulerStackTransform);

  using Self = EulerStackTransform;
  using Superclass = itk::StackTransform<CoordRepType, NDimension, NDimension>;
  using Pointer = itk::SmartPointer<EulerStackTransform>;
  itkNewMacro(Self);
  itkTypeMacro(EulerStackTransform, Superclass);

protected:
  /** Default-constructor */
  EulerStackTransform() = default;

  /** Destructor */
  ~EulerStackTransform() override = default;

private:
  /** Create a subtransform that may be added to this specific stack. */
  typename Superclass::SubTransformPointer
  CreateSubTransform() const override
  {
    return EulerTransform<CoordRepType, NDimension - 1>::New().GetPointer();
  }
};

} // namespace itk

#endif
