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
#ifndef __itkUseMevisDicomTiff_cxx
#define __itkUseMevisDicomTiff_cxx

#include "itkUseMevisDicomTiff.h"

/** avoid dependencies when not using mevis dicom tiff.
 * Also in CMakeList, only include the .cxx files when needed. */
#ifdef _ELASTIX_USE_MEVISDICOMTIFF
  #include "itkMevisDicomTiffImageIOFactory.h"
  #include "itkObjectFactoryBase.h"
#endif

/** Function that registers the Mevis DicomTiff IO factory.
 *  Call this in your program, before you load/write any images. */
void
RegisterMevisDicomTiff( void )
{
#ifdef _ELASTIX_USE_MEVISDICOMTIFF
  itk::ObjectFactoryBase::RegisterFactory( itk::MevisDicomTiffImageIOFactory::New(),
    itk::ObjectFactoryBase::INSERT_AT_FRONT );
#endif
}


#endif
