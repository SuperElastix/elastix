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
#ifndef elxDefaultConstruct_h
#define elxDefaultConstruct_h

#include <itkLightObject.h>
#include <cassert>

namespace elastix
{
/// Allows default-constructing an `itk::LightObject` derived object without calling `New()`.
/// May improve the runtime performance, by avoiding heap allocation and pointer indirection.
///
/// Follows C++ Core Guidelines, September 23, 2022, "Prefer scoped objects, don't heap-allocate unnecessarily", from
/// http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rr-scoped
///
/// \note While `New()` may use a factory (`itk::ObjectFactory`) to create the object, `DefaultConstruct` just
/// default-constructs the object.
template <typename TObject>
class DefaultConstruct : public TObject
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(DefaultConstruct);

  /// Public default-constructor. Just calls the (typically protected) default-constructor of `TObject`.
  DefaultConstruct() = default;

  /// Public destructor. Just calls the (typically protected) destructor of `TObject`.
  ~DefaultConstruct() override
  {
    // Note: This assertion may fail when a filter is default-constructed _before_ its input, e.g.:
    //
    // {
    //   DefaultConstruct<itk::TransformixFilter<ImageType>> transformixFilter{};
    //   DefaultConstruct<ImageType>                         movingImage{};
    //   transformixFilter.SetMovingImage(&movingImage);
    // } // <== Assertion failure!
    //
    // Such a failure may be solved by reordering the declarations (declaring the input first), or by declaring the
    // input by their static `New()` member function (instead of by `DefaultConstruct`).
    assert(this->itk::LightObject::m_ReferenceCount == 1);

    // Suppress warning "Trying to delete object with non-zero reference count."
    this->itk::LightObject::m_ReferenceCount = 0;
  }
};
} // namespace elastix

#endif
