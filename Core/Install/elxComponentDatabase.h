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

#ifndef elxComponentDatabase_h
#define elxComponentDatabase_h

#include "itkObject.h"
#include "itkObjectFactory.h"
#include <iostream>
#include <string>
#include <utility>
#include <map>

namespace elastix
{

/**
 * \class ComponentDatabase
 *
 * \brief The ComponentDatabase class is a class that stores the
 * New() functions of all components.
 *
 * In elastix the metric/transform/dimension/pixeltype etc. are all selected
 * at runtime. To make this possible, all components (metric/transform etc)
 * have to compiled for different dimension/pixeltype. The elx::ComponentDatabase
 * stores for each instance and each pixeltype/dimension a pointers to a function
 * that creates a component of the specific type.
 *
 * Each new component (a new metric for example should "make itself
 * known" by calling the elxInstallMacro, which is defined in
 * elxMacro.h .
 *
 * \sa elxInstallFunctions
 * \ingroup Install
 */

class ComponentDatabase : public itk::Object
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ComponentDatabase);

  /** Standard.*/
  using Self = ComponentDatabase;
  using Superclass = itk::Object;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  itkNewMacro(Self);
  itkTypeMacro(ComponentDatabase, Object);

  /** The Index is the number of the ElastixTypedef<number>::ElastixType.*/
  using IndexType = unsigned int;

  /** Typedefs for the CreatorMap*/
  using ObjectPointer = itk::Object::Pointer;

  /** PtrToCreator is a pointer to a function which
   * outputs an ObjectPointer and has no input arguments.
   */
  using PtrToCreator = ObjectPointer (*)();
  using ComponentDescriptionType = std::string;
  using CreatorMapKeyType = std::pair<ComponentDescriptionType, IndexType>;
  using CreatorMapValueType = PtrToCreator;
  using CreatorMapType = std::map<CreatorMapKeyType, CreatorMapValueType>;
  using CreatorMapEntryType = CreatorMapType::value_type;

  /** Typedefs for the IndexMap.*/

  /** The ImageTypeDescription contains the pixeltype (as a string)
   * and the dimension (unsigned int).
   */
  using PixelTypeDescriptionType = std::string;
  using ImageDimensionType = unsigned int;
  using ImageTypeDescriptionType = std::pair<PixelTypeDescriptionType, ImageDimensionType>;

  /** This pair contains the ImageTypeDescription of the FixedImageType
   * and the MovingImageType.
   */
  using IndexMapKeyType = std::pair<ImageTypeDescriptionType, ImageTypeDescriptionType>;
  using IndexMapValueType = IndexType;
  using IndexMapType = std::map<IndexMapKeyType, IndexMapValueType>;
  using IndexMapEntryType = IndexMapType::value_type;

  /** Functions to set an entry in a map.*/
  int
  SetCreator(const ComponentDescriptionType & name, IndexType i, PtrToCreator creator);

  int
  SetIndex(const PixelTypeDescriptionType & fixedPixelType,
           ImageDimensionType               fixedDimension,
           const PixelTypeDescriptionType & movingPixelType,
           ImageDimensionType               movingDimension,
           IndexType                        i);

  /** Functions to get an entry in a map */
  PtrToCreator
  GetCreator(const ComponentDescriptionType & name, IndexType i) const;

  IndexType
  GetIndex(const PixelTypeDescriptionType & fixedPixelType,
           ImageDimensionType               fixedDimension,
           const PixelTypeDescriptionType & movingPixelType,
           ImageDimensionType               movingDimension) const;

protected:
  ComponentDatabase() = default;
  ~ComponentDatabase() override = default;

private:
  CreatorMapType CreatorMap;
  IndexMapType   IndexMap;
};

} // end namespace elastix

#endif // end #ifndef elxComponentDatabase_h
