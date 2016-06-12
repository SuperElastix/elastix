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

#ifndef __elxComponentDatabase_h
#define __elxComponentDatabase_h

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

class ComponentDatabase :
  public itk::Object
{
public:

  /** Standard.*/
  typedef ComponentDatabase               Self;
  typedef itk::Object                     Superclass;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  itkNewMacro( Self );
  itkTypeMacro( ComponentDatabase, Object );

  /** The Index is the number of the ElastixTypedef<number>::ElastixType.*/
  typedef unsigned int IndexType;

  /** Typedefs for the CreatorMap*/
  typedef itk::Object         ObjectType;
  typedef ObjectType::Pointer ObjectPointer;

  /** PtrToCreator is a pointer to a function which
   * outputs an ObjectPointer and has no input arguments.
   */
  typedef ObjectPointer (* PtrToCreator)( void );
  typedef std::string      ComponentDescriptionType;
  typedef std::pair<
    ComponentDescriptionType,
    IndexType >                        CreatorMapKeyType;
  typedef PtrToCreator CreatorMapValueType;
  typedef std::map<
    CreatorMapKeyType,
    CreatorMapValueType >              CreatorMapType;
  typedef CreatorMapType::value_type CreatorMapEntryType;

  /** Typedefs for the IndexMap.*/

  /** The ImageTypeDescription contains the pixeltype (as a string)
   * and the dimension (unsigned int).
   */
  typedef std::string  PixelTypeDescriptionType;
  typedef unsigned int ImageDimensionType;
  typedef std::pair<
    PixelTypeDescriptionType,
    ImageDimensionType >   ImageTypeDescriptionType;

  /** This pair contains the ImageTypeDescription of the FixedImageType
   * and the MovingImageType.
   */
  typedef std::pair<
    ImageTypeDescriptionType,
    ImageTypeDescriptionType >         IndexMapKeyType;
  typedef IndexType IndexMapValueType;
  typedef std::map<
    IndexMapKeyType,
    IndexMapValueType >                IndexMapType;
  typedef IndexMapType::value_type IndexMapEntryType;

  /** Functions to get the CreatorMap and the IndexMap.*/
  CreatorMapType & GetCreatorMap( void );

  IndexMapType & GetIndexMap( void );

  /** Functions to set an entry in a map.*/
  int SetCreator(
    const ComponentDescriptionType & name,
    IndexType i,
    PtrToCreator creator );

  int SetIndex(
    const PixelTypeDescriptionType & fixedPixelType,
    ImageDimensionType fixedDimension,
    const PixelTypeDescriptionType & movingPixelType,
    ImageDimensionType movingDimension,
    IndexType i );

  /** Functions to get an entry in a map */
  PtrToCreator GetCreator(
    const ComponentDescriptionType & name,
    IndexType i );

  IndexType GetIndex(
    const PixelTypeDescriptionType & fixedPixelType,
    ImageDimensionType fixedDimension,
    const PixelTypeDescriptionType & movingPixelType,
    ImageDimensionType movingDimension );

protected:

  ComponentDatabase(){}
  virtual ~ComponentDatabase(){}

  CreatorMapType CreatorMap;
  IndexMapType   IndexMap;

private:

  ComponentDatabase( const Self & ); // purposely not implemented
  void operator=( const Self & );    // purposely not implemented

};

} // end namespace elastix

#endif // end #ifndef __elxComponentDatabase_h
