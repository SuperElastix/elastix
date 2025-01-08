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
#ifndef itkTranslationStackTransform_h
#define itkTranslationStackTransform_h

#include "itkStackTransform.h"
#include "itkAdvancedTranslationTransform.h"
#include "elxElastixBase.h"

namespace itk
{
template <unsigned int NDimension>
class ITK_TEMPLATE_EXPORT TranslationStackTransform
  : public itk::StackTransform<elx::ElastixBase::CoordinateType, NDimension, NDimension>
{
private:
  using CoordinateType = elx::ElastixBase::CoordinateType;

public:
  ITK_DISALLOW_COPY_AND_MOVE(TranslationStackTransform);

  using Self = TranslationStackTransform;
  using Superclass = itk::StackTransform<CoordinateType, NDimension, NDimension>;
  using Pointer = itk::SmartPointer<TranslationStackTransform>;
  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(TranslationStackTransform);

protected:
  /** Default-constructor */
  TranslationStackTransform() = default;

  /** Destructor */
  ~TranslationStackTransform() override = default;

private:
  /** Create a subtransform that may be added to this specific stack. */
  typename Superclass::SubTransformPointer
  CreateSubTransform() const override
  {
    return AdvancedTranslationTransform<CoordinateType, NDimension - 1>::New().GetPointer();
  }
};

} // namespace itk

#endif
