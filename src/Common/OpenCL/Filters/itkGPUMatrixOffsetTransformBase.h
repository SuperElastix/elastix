/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUMatrixOffsetTransformBase_h
#define __itkGPUMatrixOffsetTransformBase_h

#include "itkGPUTransformBase.h"
#include "itkMatrix.h"

namespace itk
{
/** Create a helper GPU Kernel class for itkGPUMatrixOffsetTransformBase */
itkGPUKernelClassMacro( GPUMatrixOffsetTransformBaseKernel );

/** \class GPUMatrixOffsetTransformBase
* \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
* Department of Radiology, Leiden, The Netherlands
*
* This implementation was taken from elastix (http://elastix.isi.uu.nl/).
*
* \note This work was funded by the Netherlands Organisation for
* Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
*
*/
template<
  class TScalarType = float,           // Data type for scalars
  unsigned int NInputDimensions = 3,   // Number of dimensions in the input space
  unsigned int NOutputDimensions = 3 > // Number of dimensions in the output space
class ITK_EXPORT GPUMatrixOffsetTransformBase : public GPUTransformBase
{
public:
  /** Standard typedefs   */
  typedef GPUMatrixOffsetTransformBase Self;
  typedef GPUTransformBase             Superclass;
  typedef SmartPointer< Self >         Pointer;
  typedef SmartPointer< const Self >   ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUMatrixOffsetTransformBase, Superclass );

  /**  */
  virtual bool IsMatrixOffsetTransform() const { return true; }

  /** Type of the scalar representing coordinate and vector elements. */
  typedef TScalarType ScalarType;

  /** Dimension of the domain space. */
  itkStaticConstMacro( InputSpaceDimension, unsigned int, NInputDimensions );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, NOutputDimensions );
  itkStaticConstMacro( ParametersDimension, unsigned int, NOutputDimensions * ( NInputDimensions + 1 ) );

  /** Standard matrix type for this class */
  typedef Matrix< TScalarType, itkGetStaticConstMacro( OutputSpaceDimension ),
                  itkGetStaticConstMacro( InputSpaceDimension ) >  CPUMatrixType;
  typedef Matrix< TScalarType, itkGetStaticConstMacro( InputSpaceDimension ),
                  itkGetStaticConstMacro( OutputSpaceDimension ) > CPUInverseMatrixType;
  typedef Vector< TScalarType,
                  itkGetStaticConstMacro( OutputSpaceDimension ) > CPUOutputVectorType;

  /**  */
  virtual const CPUMatrixType & GetCPUMatrix( void ) const = 0;
  virtual const CPUInverseMatrixType & GetCPUInverseMatrix( void ) const = 0;
  virtual const CPUOutputVectorType & GetCPUOffset( void ) const = 0;

protected:
  GPUMatrixOffsetTransformBase();
  virtual ~GPUMatrixOffsetTransformBase() {}

  virtual bool GetSourceCode( std::string & _source ) const;

  virtual GPUDataManager::Pointer GetParametersDataManager() const;

private:
  GPUMatrixOffsetTransformBase( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );             // purposely not implemented

  std::vector< std::string > m_Sources;
  bool                       m_SourcesLoaded;
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUMatrixOffsetTransformBase.hxx"
#endif

#endif /* itkGPUMatrixOffsetTransformBase_h */
