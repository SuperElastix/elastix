/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#include "elxTestOutputWindow.h"

namespace itk
{
//------------------------------------------------------------------------------
void TestOutputWindow::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
}

//------------------------------------------------------------------------------
void TestOutputWindow::DisplayText( const char *text )
{
  std::cout << "ITK " << text;
}

//------------------------------------------------------------------------------
void TestOutputWindow::DisplayErrorText( const char *text )
{
  std::cout << "ITK " << text;
}

//------------------------------------------------------------------------------
void TestOutputWindow::DisplayWarningText( const char *text )
{
  std::cout << "ITK " << text;
}

//------------------------------------------------------------------------------
void TestOutputWindow::DisplayGenericOutputText( const char *text )
{
  std::cout << "ITK " << text;
}

//------------------------------------------------------------------------------
void TestOutputWindow::DisplayDebugText( const char *text )
{
  std::cout << "ITK " << text;
}
} // end namespace itk
