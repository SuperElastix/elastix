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

// First include the header file to be tested:
#include "elxDefaultConstruct.h"
#include <itkImage.h>
#include <gtest/gtest.h>
#include <type_traits> // For is_base_of and is_default_constructible.

// The class template to be tested:
using elx::DefaultConstruct;

// Example type, to be used as template argument of DefaultConstruct.
using ImageType = itk::Image<int>;

namespace
{
// A minimal test class, to be used as template argument of DefaultConstruct.
class TestObject : public itk::LightObject
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TestObject);
  using Self = TestObject;
  using Superclass = itk::LightObject;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro(Self);

protected:
  TestObject() = default;
  ~TestObject() = default;

private:
  int m_data{};

  friend bool
  operator==(const Self & lhs, const Self & rhs)
  {
    return lhs.m_data == rhs.m_data;
  }

  friend bool
  operator!=(const Self & lhs, const Self & rhs)
  {
    return !(lhs == rhs);
  }
};
} // namespace


static_assert(std::is_base_of<TestObject, DefaultConstruct<TestObject>>{} &&
                std::is_base_of<ImageType, DefaultConstruct<ImageType>>{},
              "DefaultConstruct<T> must be a subclass of T! ");

static_assert(std::is_default_constructible<DefaultConstruct<TestObject>>{} &&
                std::is_default_constructible<DefaultConstruct<ImageType>>{},
              "DefaultConstruct<T> must be default-constructible! ");

GTEST_TEST(DefaultConstruct, Check)
{
  const DefaultConstruct<ImageType>  defaultConstructedImage{};
  const DefaultConstruct<TestObject> defaultConstructedTestObject{};

  EXPECT_EQ(defaultConstructedTestObject, *TestObject::New());
  EXPECT_EQ(defaultConstructedImage, *ImageType::New());
}
