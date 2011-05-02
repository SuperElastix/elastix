/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#include "itkMevisDicomTiffImageIOFactory.h"
#include "itkCreateObjectFunction.h"
#include "itkMevisDicomTiffImageIO.h"
#include "itkVersion.h"


namespace itk
{
MevisDicomTiffImageIOFactory::MevisDicomTiffImageIOFactory()
{
  this->RegisterOverride("itkImageIOBase",
                         "itkMevisDicomTiffImageIO",
                         "Mevis Dicom/TIFF Image IO",
                         1,
                         CreateObjectFunction<MevisDicomTiffImageIO>::New());
}

MevisDicomTiffImageIOFactory::~MevisDicomTiffImageIOFactory()
{
}

const char*
MevisDicomTiffImageIOFactory::GetITKSourceVersion() const
{
  return ITK_SOURCE_VERSION;
}

const char*
MevisDicomTiffImageIOFactory::GetDescription() const
{
  return "Mevis Dicom/TIFF ImageIO Factory, allows the loading of Mevis images into insight";
}

} // end namespace itk
