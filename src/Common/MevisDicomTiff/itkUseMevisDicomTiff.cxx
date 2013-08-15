/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkUseMevisDicomTiff_cxx
#define __itkUseMevisDicomTiff_cxx

#include "itkUseMevisDicomTiff.h"

#include "itkMevisDicomTiffImageIOFactory.h"
#include "itkObjectFactoryBase.h"

/** Function that registers the Mevis DicomTiff IO factory.
 *  Call this in your program, before you load/write any images. */
void RegisterMevisDicomTiff(void)
{
#ifdef _ELASTIX_USE_MEVISDICOMTIFF
  itk::ObjectFactoryBase::RegisterFactory( itk::MevisDicomTiffImageIOFactory::New(),
    itk::ObjectFactoryBase::INSERT_AT_FRONT );
#endif
}


#endif
