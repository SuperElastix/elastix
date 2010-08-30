#ifndef __itkUseMevisDicomTiff_h
#define __itkUseMevisDicomTiff_h

/** Include this file in your main source code */

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include "itkUseMevisDicomTiff.h"
#include "itkMevisDicomTiffImageIOFactory.h"
#include "itkObjectFactoryBase.h"

/** Function that registers the Mevis DicomTiff IO factory. */
int __UseMevisDicomTiff(void)
{
  static bool firsttime = true;
  if (firsttime)
  {
    itk::ObjectFactoryBase::RegisterFactory( itk::MevisDicomTiffImageIOFactory::New() );
    firsttime=false;
  }
  return 0;
}

/** Dummy return variable, to call the UseMevisDicomTiff function. */
int __UseMevisDicomTiffDummy = __UseMevisDicomTiff();

#endif
