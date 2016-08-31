/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
//
// \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
// Department of Radiology, Leiden, The Netherlands
//
// \note This work was funded by the Netherlands Organisation for
// Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
//
#ifndef __itkTestOutputWindow_h
#define __itkTestOutputWindow_h

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
DisplayDebugText - <Debug>

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
  virtual void DisplayText( const char * ) ITK_OVERRIDE;

protected:

  TestOutputWindow() {}
  virtual ~TestOutputWindow() {}
  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

private:

  TestOutputWindow( const Self & ); //purposely not implemented
  void operator=( const Self & );   //purposely not implemented

};

} // end namespace itk

#endif // __itkTestOutputWindow_h
