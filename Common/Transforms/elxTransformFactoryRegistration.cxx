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

#include "elxTransformFactoryRegistration.h"

#include <../Components/Transforms/AffineLogStackTransform/itkAffineLogStackTransform.h>
#include <../Components/Transforms/BSplineStackTransform/itkBSplineStackTransform.h>
#include <../Components/Transforms/EulerStackTransform/itkEulerStackTransform.h>
#include <../Components/Transforms/TranslationStackTransform/itkTranslationStackTransform.h>

#include "elxSupportedImageDimensions.h"

#include <itkBSplineTransform.h>
#include <itkTransformFactory.h>

namespace
{
struct EmptyStruct
{};


template <template <unsigned> class TTransform, std::size_t... VDimension>
static void
RegisterTransformForEachDimension(std::index_sequence<VDimension...>)
{
  struct EmptyStruct
  {};

  const EmptyStruct registered[] = { (itk::TransformFactory<TTransform<VDimension>>::RegisterTransform(),
                                      EmptyStruct())... };
  (void)registered;
}


template <template <unsigned> class TTransform>
void
RegisterTransform()
{
  RegisterTransformForEachDimension<TTransform>(elx::SupportedFixedImageDimensionSequence);
}

template <template <unsigned> class... TTransform>
void
RegisterTransforms()
{
  const EmptyStruct registered[] = { (RegisterTransform<TTransform>(), EmptyStruct())... };
  (void)registered;
}

template <unsigned int NDimension>
using ItkBSplineTransformOrder1Type = itk::BSplineTransform<double, NDimension, 1>;
template <unsigned int NDimension>
using ItkBSplineTransformOrder2Type = itk::BSplineTransform<double, NDimension, 2>;

} // namespace

namespace elastix
{
void
TransformFactoryRegistration::RegisterTransforms()
{
  // Use C++11 "magic statics" to ensure that ::RegisterTransforms is called only once, thread-safely.
  const static EmptyStruct emptyStruct = (::RegisterTransforms<ItkBSplineTransformOrder1Type,
                                                               ItkBSplineTransformOrder2Type,
                                                               itk::AffineLogStackTransform,
                                                               itk::BSplineStackTransform,
                                                               itk::EulerStackTransform,
                                                               itk::TranslationStackTransform>(),
                                          EmptyStruct());
  (void)emptyStruct;
}

} // namespace elastix
