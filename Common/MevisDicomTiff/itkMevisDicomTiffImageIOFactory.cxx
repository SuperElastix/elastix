/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMevisDicomTiffImageIOFactory.cxx,v $
  Language:  C++
  Date:      $Date: 2008/12/07 16:38:38 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
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
