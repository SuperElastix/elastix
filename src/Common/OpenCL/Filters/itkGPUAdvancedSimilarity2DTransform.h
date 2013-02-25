/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUAdvancedSimilarity2DTransform_h
#define __itkGPUAdvancedSimilarity2DTransform_h

#include "itkAdvancedSimilarity2DTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"

namespace itk
{
/** \class GPUAdvancedSimilarity2DTransform
* \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
* Department of Radiology, Leiden, The Netherlands
*
* This implementation was taken from elastix (http://elastix.isi.uu.nl/).
*
* \note This work was funded by the Netherlands Organisation for
* Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
*
*/
template< class TScalarType = float,
          class TParentImageFilter = AdvancedSimilarity2DTransform< TScalarType > >
class GPUAdvancedSimilarity2DTransform :
  public TParentImageFilter,
  public GPUMatrixOffsetTransformBase< TScalarType, 2, 2 >
{
public:
  /** Standard class typedefs. */
  typedef GPUAdvancedSimilarity2DTransform                  Self;
  typedef TParentImageFilter                                CPUSuperclass;
  typedef GPUMatrixOffsetTransformBase< TScalarType, 2, 2 > GPUSuperclass;
  typedef SmartPointer< Self >                              Pointer;
  typedef SmartPointer< const Self >                        ConstPointer;

  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAdvancedSimilarity2DTransform, CPUSuperclass );

  /** Typedefs */
  typedef typename GPUSuperclass::CPUMatrixType        CPUMatrixType;
  typedef typename GPUSuperclass::CPUInverseMatrixType CPUInverseMatrixType;
  typedef typename GPUSuperclass::CPUOutputVectorType  CPUOutputVectorType;

  /**  */
  virtual const CPUMatrixType &        GetCPUMatrix( void ) const { return this->GetMatrix(); }
  virtual const CPUInverseMatrixType & GetCPUInverseMatrix( void ) const { return this->GetInverseMatrix(); }
  virtual const CPUOutputVectorType &  GetCPUOffset( void ) const { return this->GetOffset(); }

protected:
  GPUAdvancedSimilarity2DTransform() {}
  virtual ~GPUAdvancedSimilarity2DTransform() {}

private:
  GPUAdvancedSimilarity2DTransform( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );                 // purposely not implemented
};

/** \class GPUAdvancedSimilarity2DTransformFactory
* \brief Object Factory implementation for GPUAdvancedSimilarity2DTransform
*/
class GPUAdvancedSimilarity2DTransformFactory : public ObjectFactoryBase
{
public:
  typedef GPUAdvancedSimilarity2DTransformFactory Self;
  typedef ObjectFactoryBase                       Superclass;
  typedef SmartPointer< Self >                    Pointer;
  typedef SmartPointer< const Self >              ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char * GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
  const char * GetDescription() const { return "A Factory for GPUAdvancedSimilarity2DTransform"; }

  /** Method for class instantiation. */
  itkFactorylessNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUAdvancedSimilarity2DTransformFactory, itk::ObjectFactoryBase );

  /** Register one factory of this type  */
  static void RegisterOneFactory( void )
  {
    GPUAdvancedSimilarity2DTransformFactory::Pointer factory =
      GPUAdvancedSimilarity2DTransformFactory::New();
    ObjectFactoryBase::RegisterFactory( factory );
  }

private:
  GPUAdvancedSimilarity2DTransformFactory( const Self & ); // purposely not implemented
  void operator=( const Self & );                          // purposely not implemented

#define OverrideAdvancedSimilarity2DTransformTypeMacro( st )                   \
  {                                                                            \
    this->RegisterOverride(                                                    \
      typeid( AdvancedSimilarity2DTransform< st > ).name(),                    \
      typeid( GPUAdvancedSimilarity2DTransform< st > ).name(),                 \
      "GPU AdvancedSimilarity2DTransform Override",                            \
      true,                                                                    \
      CreateObjectFunction< GPUAdvancedSimilarity2DTransform< st > >::New() ); \
  }

  GPUAdvancedSimilarity2DTransformFactory()
  {
    if ( IsGPUAvailable() )
    {
      OverrideAdvancedSimilarity2DTransformTypeMacro( float );
      //OverrideAdvancedSimilarity2DTransformTypeMacro( double );

      OverrideAdvancedSimilarity2DTransformTypeMacro( float );
      //OverrideAdvancedSimilarity2DTransformTypeMacro( double );

      OverrideAdvancedSimilarity2DTransformTypeMacro( float );
      //OverrideAdvancedSimilarity2DTransformTypeMacro( double );
    }
  }
};
} // end namespace itk

#endif /* __itkGPUAdvancedSimilarity2DTransform_h */
