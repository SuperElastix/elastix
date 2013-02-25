/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxTestOutputWindow_h
#define __elxTestOutputWindow_h

// ITK include files
#include "itkOutputWindow.h"
#include "itkObjectFactory.h"

namespace itk
{
//! Definition of TestOutputWindow.
/*! TestOutputWindow - test itk specific output window class
Writes debug/warning/error output to std::cout.
The text is processed to replace:

DisplayText - <Text>
DisplayErrorText - <Error>
DisplayWarningText - <Warning>
DisplayGenericWarningText - <GenericWarning>
DisplayDebugText - <Debug.

\sa OutputWindow
*/
class TestOutputWindow : public OutputWindow
{
public:
  /** Standard class typedefs. */
  typedef TestOutputWindow           Self;
  typedef OutputWindow               Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( TestOutputWindow, OutputWindow );

  /** Overloaded */
  virtual void DisplayText( const char * );

  virtual void DisplayErrorText( const char * );

  virtual void DisplayWarningText( const char * );

  virtual void DisplayGenericOutputText( const char * );

  virtual void DisplayDebugText( const char * );

protected:
  TestOutputWindow() {}
  virtual ~TestOutputWindow() {}
  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

private:
  TestOutputWindow( const Self & ); //purposely not implemented
  void operator=( const Self & );   //purposely not implemented
};
} // end namespace itk

#endif // __elxTestOutputWindow_h
