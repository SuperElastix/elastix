/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkUseMevisDicomTiff_h
#define __itkUseMevisDicomTiff_h

/** Include this file in your main source code */

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

/** Function that registers the Mevis DicomTiff IO factory.
 *  Call this in your program, before you load/write any images. */
void RegisterMevisDicomTiff( void );

#endif
