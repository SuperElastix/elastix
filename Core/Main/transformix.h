/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __transformix_h
#define __transformix_h

#include "itkUseMevisDicomTiff.h"

#include "elxTransformixMain.h"
#include <iostream>
#include <string>
#include <vector>
#include <queue>

#include "itkObject.h"
#include "itkDataObject.h"
#include <itksys/SystemTools.hxx>
#include <itksys/SystemInformation.hxx>

#include "elxTimer.h"

/** Declare PrintHelp function.*/
void PrintHelp( void );

#endif // end #ifndef __transformix_h
