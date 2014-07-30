/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxDeformationFieldTransform_HXX__
#define __elxDeformationFieldTransform_HXX__

#include "elxDeformationFieldTransform.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkVectorNearestNeighborInterpolateImageFunction.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkChangeInformationImageFilter.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template< class TElastix >
DeformationFieldTransform< TElastix >
::DeformationFieldTransform()
{
  /** Initialize. */
  this->m_DeformationFieldInterpolatingTransform
    = DeformationFieldInterpolatingTransformType::New();
  this->SetCurrentTransform(
    this->m_DeformationFieldInterpolatingTransform );

  /** Make sure that the TransformBase::WriteToFile() does
   * not read the transformParameters in the file. */
  this->SetReadWriteTransformParameters( false );

  /** Initialize to identity. */
  this->m_OriginalDeformationFieldDirection.SetIdentity();

}   // end Constructor


/**
 * ************************* ReadFromFile ************************
 */

template< class TElastix >
void
DeformationFieldTransform< TElastix >::ReadFromFile( void )
{
  // \todo Test this ReadFromFile function.

  /** Call the ReadFromFile from the TransformBase. */
  this->Superclass2::ReadFromFile();

  typedef itk::ChangeInformationImageFilter< DeformationFieldType > ChangeInfoFilterType;
  typedef typename ChangeInfoFilterType::Pointer                    ChangeInfoFilterPointer;

  /** Setup VectorImageReader. */
  typedef itk::ImageFileReader< DeformationFieldType > VectorReaderType;
  typename VectorReaderType::Pointer vectorReader
    = VectorReaderType::New();

  /** Read deformationFieldImage-name from parameter-file. */
  std::string fileName = "";
  this->m_Configuration->ReadParameter( fileName,
    "DeformationFieldFileName", 0 );
  if( fileName == "" )
  {
    xl::xout[ "error" ] << "ERROR: the entry (DeformationFieldFileName \"...\") is missing in the transform parameter file!" << std::endl;
    itkExceptionMacro( << "Error while reading transform parameter file!" );
  }

  /** Possibly overrule the direction cosines. */
  ChangeInfoFilterPointer infoChanger = ChangeInfoFilterType::New();
  DirectionType           direction;
  direction.SetIdentity();
  infoChanger->SetOutputDirection( direction );
  infoChanger->SetChangeDirection( !this->GetElastix()->GetUseDirectionCosines() );
  infoChanger->SetInput( vectorReader->GetOutput() );

  /** Read deformationFieldImage from file. */
  vectorReader->SetFileName( fileName.c_str() );
  try
  {
    infoChanger->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "DeformationFieldTransform - ReadFromFile()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occured while reading the deformationField image.\n";
    excp.SetDescription( err_str );
    /** Pass the exception to an higher level. */
    throw excp;
  }

  /** Store the original direction for later use */
  this->m_OriginalDeformationFieldDirection
    = vectorReader->GetOutput()->GetDirection();

  /** Set the deformationFieldImage in the
   * itkDeformationFieldInterpolatingTransform.
   */
  this->m_DeformationFieldInterpolatingTransform->
  SetDeformationField( infoChanger->GetOutput() );

  typedef typename DeformationFieldInterpolatingTransformType::
    DeformationFieldInterpolatorType InterpolatorType;
  typedef itk::VectorNearestNeighborInterpolateImageFunction<
    DeformationFieldType, CoordRepType >  NNInterpolatorType;
  typedef itk::VectorLinearInterpolateImageFunction<
    DeformationFieldType, CoordRepType >  LinInterpolatorType;

  typename InterpolatorType::Pointer interpolator = 0;
  unsigned int interpolationOrder = 0;
  this->m_Configuration->ReadParameter( interpolationOrder,
    "DeformationFieldInterpolationOrder", 0 );
  if( interpolationOrder == 0 )
  {
    interpolator = NNInterpolatorType::New();
  }
  else if( interpolationOrder == 1 )
  {
    interpolator = LinInterpolatorType::New();
  }
  else
  {
    xl::xout[ "error" ] << "Error while reading DeformationFieldInterpolationOrder from the parameter file" << std::endl;
    xl::xout[ "error" ] << "DeformationFieldInterpolationOrder can only be 0 or 1!" << std::endl;
    itkExceptionMacro( << "Invalid deformation field interpolation order selected!" );
  }
  this->m_DeformationFieldInterpolatingTransform->
  SetDeformationFieldInterpolator( interpolator );

}   // end ReadFromFile()


/**
 * ************************* WriteToFile ************************
 *
 * Saves the TransformParameters as a vector and if wanted
 * also as a deformation field.
 */

template< class TElastix >
void
DeformationFieldTransform< TElastix >
::WriteToFile( const ParametersType & param ) const
{
  // \todo Finish and Test this WriteToFile function.

  /** Call the WriteToFile from the TransformBase. */
  this->Superclass2::WriteToFile( param );

  typedef itk::ChangeInformationImageFilter< DeformationFieldType > ChangeInfoFilterType;

  /** Add some DeformationFieldTransform specific lines. */
  xout[ "transpar" ] << std::endl << "// DeformationFieldTransform specific" << std::endl;

  /** Get the last part of the filename of the transformParameter-file,
   * which is going to be part of the filename of the deformationField image.
   */
  std::string                          ctpfn    = this->GetElastix()->GetCurrentTransformParameterFileName();
  std::basic_string< char >::size_type pos      = ctpfn.rfind( "TransformParameters." );
  std::string                          lastpart = ctpfn.substr( pos + 19, ctpfn.size() - pos - 19 - 4 );

  /** Create the filename of the deformationField image. */
  std::string resultImageFormat = "mhd";
  this->m_Configuration->ReadParameter( resultImageFormat, "ResultImageFormat", 0, false );
  std::ostringstream makeFileName( "" );
  makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
               << "DeformationFieldImage"
               << lastpart
               << "." << resultImageFormat;
  xout[ "transpar" ] << "(DeformationFieldFileName \""
                     << makeFileName.str() << "\")" << std::endl;

  /** Write the interpolation order to file */
  std::string interpolatorName
    = this->m_DeformationFieldInterpolatingTransform->
    GetDeformationFieldInterpolator()->GetNameOfClass();

  unsigned int interpolationOrder = 0;
  if( interpolatorName == "NearestNeighborInterpolateImageFunction" )
  {
    interpolationOrder = 0;
  }
  else if( interpolatorName == "LinearInterpolateImageFunction" )
  {
    interpolationOrder = 1;
  }
  xout[ "transpar" ] << "(DeformationFieldInterpolationOrder "
                     <<  interpolationOrder << ")" << std::endl;

  /** Possibly change the direction cosines to there original value */
  typename ChangeInfoFilterType::Pointer infoChanger = ChangeInfoFilterType::New();
  infoChanger->SetOutputDirection( this->m_OriginalDeformationFieldDirection );
  infoChanger->SetChangeDirection( !this->GetElastix()->GetUseDirectionCosines() );
  infoChanger->SetInput( this->m_DeformationFieldInterpolatingTransform->GetDeformationField() );

  /** Write the deformation field image. */
  typedef itk::ImageFileWriter< DeformationFieldType > VectorWriterType;
  typename VectorWriterType::Pointer writer
    = VectorWriterType::New();
  writer->SetFileName( makeFileName.str().c_str() );
  writer->SetInput( infoChanger->GetOutput() );

  /** Do the writing. */
  try
  {
    writer->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "DeformationFieldTransform - WriteToFile()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError while writing the deformationFieldImage.\n";
    excp.SetDescription( err_str );
    /** Print the exception. */
    xl::xout[ "error" ] << excp << std::endl;
  }

}   // end WriteToFile()


} // end namespace elastix

#endif // end #ifndef __elxDeformationFieldTransform_HXX__
