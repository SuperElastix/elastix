/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMevisDicomTiffImageIOFactory.cxx,v $
  Language:  C++
  Date:      $Date: 2009/10/14 13:28:12 $
  Version:   $Revision: 1.7 $

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
