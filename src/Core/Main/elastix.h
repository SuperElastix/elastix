/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elastix_h
#define __elastix_h

#include "itkUseMevisDicomTiff.h"

#include "elxElastixMain.h"
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include "itkObject.h"
#include "itkDataObject.h"
#include <itksys/SystemTools.hxx>
#include <itksys/SystemInformation.hxx>

#include "elxTimer.h"

  /** Declare PrintHelp function.
   *
   * \commandlinearg --help: optional argument for elastix and transformix to call the help. \n
   *    example: <tt>elastix --help</tt> \n
   *    example: <tt>transformix --help</tt> \n
   * \commandlinearg --version: optional argument for elastix and transformix to call
   *    version information. \n
   *    example: <tt>elastix --version</tt> \n
   *    example: <tt>transformix --version</tt> \n
   */
  void PrintHelp(void);

#endif
