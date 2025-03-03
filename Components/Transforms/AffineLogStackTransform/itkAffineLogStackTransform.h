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
#ifndef itkAffineLogStackTransform_h
#define itkAffineLogStackTransform_h

#include "itkStackTransform.h"
#include "../AffineLogTransform/itkAffineLogTransform.h"
#include "elxElastixBase.h"

namespace itk
{
template <unsigned int NDimension>
class ITK_TEMPLATE_EXPORT AffineLogStackTransform
  : public itk::StackTransform<elx::ElastixBase::CoordinateType, NDimension, NDimension>
{
private:
  using CoordinateType = elx::ElastixBase::CoordinateType;

public:
  ITK_DISALLOW_COPY_AND_MOVE(AffineLogStackTransform);

  using Self = AffineLogStackTransform;
  using Superclass = itk::StackTransform<CoordinateType, NDimension, NDimension>;
  using Pointer = itk::SmartPointer<AffineLogStackTransform>;
  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(AffineLogStackTransform);

protected:
  /** Default-constructor */
  AffineLogStackTransform() = default;

  /** Destructor */
  ~AffineLogStackTransform() override = default;

private:
  /** Create a subtransform that may be added to this specific stack. */
  typename Superclass::SubTransformPointer
  CreateSubTransform() const override
  {
    return AffineLogTransform<CoordinateType, NDimension - 1>::New().GetPointer();
  }
};

} // namespace itk

#endif
