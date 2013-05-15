/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __transformixlib_h
#define __transformixlib_h

/**
 *  Includes
 */
#include <itkDataObject.h>

/********************************************************************************
 *                    *
 *      Dll export    *
 *                    *
 ********************************************************************************/
#define TRFX_COMPILE_LIB 1

#if (defined(_WIN32) || defined(WIN32) ) 
# ifdef TRFX_COMPILE_LIB
#  define TRANSFORMIXLIB_API __declspec(dllexport)
# else
#  define TRANSFORMIXLIB_API __declspec(dllimport)
# endif 
#else
/* unix needs nothing */
#define ELX_EXPORT 
#endif

/********************************************************************************
 *                          *
 *      Class definition    *
 *                          *
 ********************************************************************************/
namespace transformix
{

class TRANSFORMIXLIB_API TRANSFORMIX
{
public:
  //typedefs for images
  typedef itk::DataObject   Image;
  typedef Image::Pointer    ImagePointer;

  //typedefs for parameter map
  typedef std::vector< std::string >      ParameterValuesType;
  typedef std::map< std::string, ParameterValuesType > ParameterMapType;

  /** Constructor and destructor. */
  TRANSFORMIX::TRANSFORMIX();
  TRANSFORMIX::~TRANSFORMIX();

  /** Return value: 0 is success in case not 0 an error occurred
   *    0 = success
   *    1 = error
   *   -2 = output folder does not exist
   */
  int TransformImage( ImagePointer inputImage,
    ParameterMapType & parameterMap,
    std::string outputPath,
    bool performLogging,
    bool performCout );

  /** Getter for result image. */
  ImagePointer GetResultImage( void );

private:
  ImagePointer  m_ResultImage;

}; // end class TRANSFORMIX

} // namespace transformix

#endif // end #ifndef __transformixlib_h
