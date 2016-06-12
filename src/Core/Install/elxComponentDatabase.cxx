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

/**
 * Implementation of the ComponentDatabase class.
 *
 */

#ifndef __elxComponentDatabase_cxx
#define __elxComponentDatabase_cxx

#include "elxComponentDatabase.h"
#include "xoutmain.h"

namespace elastix
{
using namespace xl;

/**
 * ******************** GetCreatorMap ***************************
 */

ComponentDatabase::CreatorMapType &
ComponentDatabase::GetCreatorMap( void )
{
  return CreatorMap;

}   // end GetCreatorMap


/**
 * ********************** GetIndexMap ***************************
 */

ComponentDatabase::IndexMapType &
ComponentDatabase::GetIndexMap( void )
{
  return IndexMap;

}   // end GetIndexMap


/**
 * *********************** SetCreator ***************************
 */

int
ComponentDatabase::SetCreator(
  const ComponentDescriptionType & name,
  IndexType i,
  PtrToCreator creator )
{
  /** Get the map */
  CreatorMapType & map = GetCreatorMap();

  /** Make a key with the input arguments */
  CreatorMapKeyType key( name, i );

  /** Check if this key has been defined already.
   * If not, insert the key + creator in the map.
   */
  if( map.count( key ) )    //==1
  {
    xout[ "error" ] << "Error: " << std::endl;
    xout[ "error" ] << name << "(index " << i << ") - This component has already been installed!" << std::endl;
    return 1;
  }
  else
  {
    map.insert( CreatorMapEntryType( key, creator ) );
    return 0;
  }

}   // end SetCreator


/**
 * *********************** SetIndex *****************************
 */

int
ComponentDatabase::SetIndex(
  const PixelTypeDescriptionType & fixedPixelType,
  ImageDimensionType fixedDimension,
  const PixelTypeDescriptionType & movingPixelType,
  ImageDimensionType movingDimension,
  IndexType i )
{
  /** Get the map.*/
  IndexMapType & map = GetIndexMap();

  /** Make a key with the input arguments.*/
  ImageTypeDescriptionType fixedImage( fixedPixelType, fixedDimension );
  ImageTypeDescriptionType movingImage( movingPixelType, movingDimension );
  IndexMapKeyType          key( fixedImage, movingImage );

  /** Insert the key+index in the map, if it hadn't been done before yet.*/
  if( map.count( key ) )  //==1
  {
    xout[ "error" ] << "Error:" << std::endl;
    xout[ "error" ] << "FixedImageType: " << fixedDimension << "D " << fixedPixelType << std::endl;
    xout[ "error" ] << "MovingImageType: " << movingDimension << "D " << movingPixelType << std::endl;
    xout[ "error" ] << "Elastix already supports this combination of ImageTypes!" << std::endl;
    return 1;
  }
  else
  {
    map.insert( IndexMapEntryType( key, i ) );
    return 0;
  }

}   // end SetIndex


/**
 * *********************** GetCreator ***************************
 */

ComponentDatabase::PtrToCreator
ComponentDatabase::GetCreator(
  const ComponentDescriptionType & name,
  IndexType i )
{
  /** Get the map */
  CreatorMapType map = GetCreatorMap();

  /** Make a key with the input arguments */
  CreatorMapKeyType key( name, i );

  /** Check if this key has been defined. If yes, return the 'creator'
   * that is linked to it.
   */
  if( map.count( key ) == 0 )    // of gewoon !map.count( key ) als boven??
  {
    xout[ "error" ] << "Error: " << std::endl;
    xout[ "error" ] << name << "(index " << i << ") - This component is not installed!" << std::endl;
    return 0;
  }
  else
  {
    return map[ key ];
  }

}   // end GetCreator


/**
 * ************************* GetIndex ***************************
 */

ComponentDatabase::IndexType
ComponentDatabase::GetIndex(
  const PixelTypeDescriptionType & fixedPixelType,
  ImageDimensionType fixedDimension,
  const PixelTypeDescriptionType & movingPixelType,
  ImageDimensionType movingDimension )
{
  /** Get the map */
  IndexMapType map = GetIndexMap();

  /** Make a key with the input arguments */
  ImageTypeDescriptionType fixedImage( fixedPixelType, fixedDimension );
  ImageTypeDescriptionType movingImage( movingPixelType, movingDimension );
  IndexMapKeyType          key( fixedImage, movingImage );

  /** Check if this key has been defined. If yes, return the 'index'
   * that is linked to it.
   */
  if( map.count( key ) == 0 )
  {
    xout[ "error" ] << "ERROR:\n"
                    << "  FixedImageType:  " << fixedDimension << "D " << fixedPixelType << std::endl
                    << "  MovingImageType: " << movingDimension << "D " << movingPixelType << std::endl
                    << "  elastix was not compiled with this combination of ImageTypes!\n"
                    << "  You have two options to solve this:\n"
                    << "  1. Add the combination to the CMake parameters ELASTIX_IMAGE_nD_PIXELTYPES and "
                    << "ELASTIX_IMAGE_DIMENSIONS, re-cmake and re-compile.\n"
                    << "  2. Change the parameters FixedInternalImagePixelType and/or MovingInternalImagePixelType "
                    << "in the elastix parameter file.\n" << std::endl;
    return 0;
  }
  else
  {
    return map[ key ];
  }

}   // end GetIndex


} // end namespace elastix

#endif // end ifndef __elxComponentDatabase_cxx
