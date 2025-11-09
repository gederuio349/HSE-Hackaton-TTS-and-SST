"""
Скрипт для запуска FastAPI приложения с HTTPS
Автоматически генерирует самоподписанный сертификат в памяти
"""
import ssl
import tempfile
import os
import ipaddress
from pathlib import Path
import uvicorn
from api import app

def generate_self_signed_cert():
    """
    Генерирует самоподписанный сертификат в памяти
    Использует библиотеку cryptography для создания сертификата
    """
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend
        import datetime
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Создаем самоподписанный сертификат
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "RU"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Moscow"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Moscow"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Local Development"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("127.0.0.1"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv4Address("10.4.0.222")),  # IP бекенда
                x509.IPAddress(ipaddress.IPv4Address("192.168.0.100")),  # IP фронтенда
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256(), default_backend())
        
        # Сохраняем во временные файлы
        cert_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pem')
        key_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pem')
        
        cert_file.write(cert.public_bytes(serialization.Encoding.PEM))
        key_file.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
        
        cert_file.close()
        key_file.close()
        
        return cert_file.name, key_file.name
        
    except ImportError:
        # Если cryptography не установлена, используем более простой метод
        print("Библиотека cryptography не найдена. Устанавливаю...")
        print("Выполните: pip install cryptography")
        raise
    except Exception as e:
        print(f"Ошибка при генерации сертификата: {e}")
        raise

def generate_simple_cert():
    """
    Альтернативный метод: использует pyOpenSSL для генерации сертификата
    """
    try:
        from OpenSSL import crypto, SSL
        import tempfile
        
        # Создаем приватный ключ
        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 2048)
        
        # Создаем самоподписанный сертификат
        cert = crypto.X509()
        cert.get_subject().C = "RU"
        cert.get_subject().ST = "Moscow"
        cert.get_subject().L = "Moscow"
        cert.get_subject().O = "Local Development"
        cert.get_subject().CN = "localhost"
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(365*24*60*60)  # 1 год
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, 'sha256')
        
        # Сохраняем во временные файлы
        cert_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pem')
        key_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pem')
        
        cert_file.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        key_file.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
        
        cert_file.close()
        key_file.close()
        
        return cert_file.name, key_file.name
        
    except ImportError:
        print("Библиотека pyOpenSSL не найдена.")
        return None, None

if __name__ == "__main__":
    # Пытаемся сгенерировать сертификат
    cert_path = None
    key_path = None
    
    try:
        cert_path, key_path = generate_self_signed_cert()
        print(f"✓ Сертификат сгенерирован: {cert_path}")
        print(f"✓ Приватный ключ сгенерирован: {key_path}")
    except Exception as e:
        print(f"Попытка альтернативного метода...")
        cert_path, key_path = generate_simple_cert()
        if cert_path and key_path:
            print(f"✓ Сертификат сгенерирован (pyOpenSSL): {cert_path}")
            print(f"✓ Приватный ключ сгенерирован: {key_path}")
        else:
            print("Ошибка: не удалось сгенерировать сертификат")
            print("Установите одну из библиотек:")
            print("  pip install cryptography")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=4010,
            ssl_keyfile=key_path,
            ssl_certfile=cert_path,
        )
    finally:
        # Удаляем временные файлы после завершения
        if cert_path and os.path.exists(cert_path):
            os.unlink(cert_path)
        if key_path and os.path.exists(key_path):
            os.unlink(key_path)

