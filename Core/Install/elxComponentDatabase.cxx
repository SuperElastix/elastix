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
 * *********************** SetCreator ***************************
 */

int
ComponentDatabase::SetCreator(
  const ComponentDescriptionType & name,
  IndexType i,
  PtrToCreator creator )
{
  /** Make a key with the input arguments */
  CreatorMapKeyType key( name, i );

  /** Check if this key has been defined already.
   * If not, insert the key + creator in the map.
   */
  if( CreatorMap.count( key ) )    //==1
  {
    xout[ "error" ] << "Error: " << std::endl;
    xout[ "error" ] << name << "(index " << i << ") - This component has already been installed!" << std::endl;
    return 1;
  }
  else
  {
    CreatorMap.insert( CreatorMapEntryType( key, creator ) );
    return 0;
  }

} // end SetCreator


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
  /** Make a key with the input arguments.*/
  ImageTypeDescriptionType fixedImage( fixedPixelType, fixedDimension );
  ImageTypeDescriptionType movingImage( movingPixelType, movingDimension );
  IndexMapKeyType          key( fixedImage, movingImage );

  /** Insert the key+index in the map, if it hadn't been done before yet.*/
  if( IndexMap.count( key ) )  //==1
  {
    xout[ "error" ] << "Error:" << std::endl;
    xout[ "error" ] << "FixedImageType: " << fixedDimension << "D " << fixedPixelType << std::endl;
    xout[ "error" ] << "MovingImageType: " << movingDimension << "D " << movingPixelType << std::endl;
    xout[ "error" ] << "Elastix already supports this combination of ImageTypes!" << std::endl;
    return 1;
  }
  else
  {
    IndexMap.insert( IndexMapEntryType( key, i ) );
    return 0;
  }

} // end SetIndex


/**
 * *********************** GetCreator ***************************
 */

ComponentDatabase::PtrToCreator
ComponentDatabase::GetCreator(
  const ComponentDescriptionType & name,
  IndexType i ) const
{
  /** Make a key with the input arguments */
  CreatorMapKeyType key( name, i );

  const auto found = CreatorMap.find(key);

  /** Check if this key has been defined. If yes, return the 'creator'
   * that is linked to it.
   */
  if( found == end(CreatorMap) )
  {
    xout[ "error" ] << "Error: " << std::endl;
    xout[ "error" ] << name << "(index " << i << ") - This component is not installed!" << std::endl;
    return 0;
  }
  else
  {
    return found->second;
  }

} // end GetCreator


/**
 * ************************* GetIndex ***************************
 */

ComponentDatabase::IndexType
ComponentDatabase::GetIndex(
  const PixelTypeDescriptionType & fixedPixelType,
  ImageDimensionType fixedDimension,
  const PixelTypeDescriptionType & movingPixelType,
  ImageDimensionType movingDimension ) const
{
  /** Make a key with the input arguments */
  ImageTypeDescriptionType fixedImage( fixedPixelType, fixedDimension );
  ImageTypeDescriptionType movingImage( movingPixelType, movingDimension );
  IndexMapKeyType          key( fixedImage, movingImage );

  const auto found = IndexMap.find(key);

  /** Check if this key has been defined. If yes, return the 'index'
   * that is linked to it.
   */
  if( found == end(IndexMap) )
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
    return found->second;
  }

} // end GetIndex


} // end namespace elastix

#endif // end ifndef __elxComponentDatabase_cxx
