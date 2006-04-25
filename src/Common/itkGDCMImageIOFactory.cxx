/**
* This class is an adapted version of the equally named file
* in the itk-tree. However, instead of loading the GDCM dicom
* reader, it loads the DICOMIO2 reader, to avoid linking the
* the gdcm library, which causes errors in Elastix. Anyway,
* elastix does not support Dicom directories.
* 
* The original ITK-copyright notice is stated here:
*/


/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "itkGDCMImageIOFactory.h"
#include "itkCreateObjectFunction.h"
#include "itkDICOMImageIO2.h"
#include "itkVersion.h"

  
namespace itk
{
GDCMImageIOFactory::GDCMImageIOFactory()
{
  /** Hack: load the DICOMImageIO instead of the GDCMImageIO.
   * \todo I'm not sure what actually happens when you try to read
   * a Dicom file... */
  this->RegisterOverride("itkImageIOBase",
                         "itkGDCMImageIO",
                         "GDCM Image IO",
                         1,
                         CreateObjectFunction<DICOMImageIO2>::New());
}
  
GDCMImageIOFactory::~GDCMImageIOFactory()
{
}

const char* GDCMImageIOFactory::GetITKSourceVersion() const
{
  return ITK_SOURCE_VERSION;
}

const char* GDCMImageIOFactory::GetDescription() const
{
  return "GDCM ImageIO Factory, allows the loading of DICOM images into Insight";
}

} // end namespace itk

