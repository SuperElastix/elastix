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

namespace elastix
{
/// Allows default-constructing an `itk::LightObject` derived object without calling `New()`.
/// May improve the runtime performance, by avoiding heap allocation and pointer indirection.
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
    // Suppress warning "Trying to delete object with non-zero reference count."
    this->itk::LightObject::m_ReferenceCount = 0;
  }
};
} // namespace elastix

#endif
