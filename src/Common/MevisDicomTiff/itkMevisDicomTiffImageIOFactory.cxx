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
#include "itkMevisDicomTiffImageIOFactory.h"
#include "itkCreateObjectFunction.h"
#include "itkMevisDicomTiffImageIO.h"
#include "itkVersion.h"

namespace itk
{

MevisDicomTiffImageIOFactory
::MevisDicomTiffImageIOFactory()
{
  this->RegisterOverride( "itkImageIOBase",
    "itkMevisDicomTiffImageIO",
    "Mevis Dicom/TIFF Image IO",
    1,
    CreateObjectFunction< MevisDicomTiffImageIO >::New() );
}


MevisDicomTiffImageIOFactory
::~MevisDicomTiffImageIOFactory()
{}

const char *
MevisDicomTiffImageIOFactory
::GetITKSourceVersion( void ) const
{
  return ITK_SOURCE_VERSION;
}


const char *
MevisDicomTiffImageIOFactory
::GetDescription( void ) const
{
  return "Mevis Dicom/TIFF ImageIO Factory, allows the loading of Mevis images into insight";
}


} // end namespace itk
